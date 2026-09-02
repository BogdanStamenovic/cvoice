#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cvoice installer. Runs on Linux, macOS and Windows.

Python rather than shell because Windows has no bash. Invoked directly or by
`ownbox install cvoice`; ownbox runs setup with stdin/stdout attached, so the
role prompt works. Non-interactive callers pass --role or set CVOICE_ROLE.
"""
from __future__ import annotations

import argparse
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = platform.system() == "Windows"
MAC = platform.system() == "Darwin"

B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
if WIN and not os.environ.get("WT_SESSION"):
    B = D = G = Y = R = X = ""


def say(msg=""): print(msg, flush=True)
def ok(msg): say(f"{G}✓{X} {msg}")
def warn(msg): say(f"{Y}!{X} {msg}")
def die(msg): say(f"{R}✗ {msg}{X}"); sys.exit(1)


def config_dir() -> Path:
    if os.environ.get("CVOICE_CONFIG_DIR"):
        return Path(os.environ["CVOICE_CONFIG_DIR"])
    if WIN:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "cvoice"
    return Path.home() / ".config/cvoice"


def bin_dir() -> Path:
    if os.environ.get("OWNBOX_BIN_DIR"):
        return Path(os.environ["OWNBOX_BIN_DIR"])
    if WIN:
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "cvoice/bin"
    return Path.home() / ".local/bin"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


def venv_script(venv: Path, name: str) -> Path:
    return venv / ("Scripts/%s.exe" % name if WIN else "bin/%s" % name)


def run(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)


def prompt_role(default="client") -> str:
    if not sys.stdin.isatty():
        role = os.environ.get("CVOICE_ROLE", default)
        warn(f"nije interaktivno — uloga: {role}")
        return role
    say()
    say(f"{B}cvoice{X} — kloniranje glasa, samostalno hostovano")
    say()
    say(f"  {B}1) klijent{X}  {D}— reci / cvoice ovde; govori sa udaljenim serverom{X}")
    say(f"  {B}2) server{X}   {D}— drži model i profile; treba GPU (ili strpljenje){X}")
    say(f"  {B}3) oba{X}      {D}— sve na jednoj mašini{X}")
    say()
    pick = input("Šta instaliraš? [1] ").strip() or "1"
    return {"1": "client", "klijent": "client", "client": "client",
            "2": "server", "server": "server",
            "3": "both", "oba": "both", "both": "both"}.get(pick.lower()) or die(
                f"nevažeći izbor: {pick}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Install cvoice")
    ap.add_argument("--role", choices=["client", "server", "both"])
    ap.add_argument("--server-url"); ap.add_argument("--token")
    ap.add_argument("--bind"); ap.add_argument("--port", type=int)
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        die("treba Python 3.10+")

    role = args.role or os.environ.get("CVOICE_ROLE") or prompt_role()
    say(f"{D}· uloga: {role}{X}")

    venv = ROOT / ".venv"
    if not venv_python(venv).exists():
        say(f"{D}· pravim venv…{X}")
        if run([sys.executable, "-m", "venv", str(venv)]).returncode != 0:
            die("ne mogu da napravim venv")
    vpy = str(venv_python(venv))
    run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    extras = "" if role == "client" else "[server]"
    say(f"{D}· instaliram paket…{X}")
    if run([vpy, "-m", "pip", "install", "-q", "-e", f".{extras}"], cwd=ROOT).returncode != 0:
        die("pip install nije uspeo")

    if role != "client":
        # torch is not a declared dependency: the correct wheel depends on the
        # CUDA the box actually has, and the default index often yields a CPU
        # build on a GPU machine. Install it explicitly and let it be overridden.
        if run([vpy, "-c", "import torch"], stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL).returncode != 0:
            idx = os.environ.get("CVOICE_TORCH_INDEX")
            if not idx and not (MAC or WIN):
                idx = "https://download.pytorch.org/whl/cu128"
            cmd = [vpy, "-m", "pip", "install", "-q", "torch", "torchaudio"]
            if idx:
                cmd += ["--index-url", idx]
            say(f"{D}· instaliram torch{' (' + idx + ')' if idx else ''}…{X}")
            if run(cmd).returncode != 0:
                warn("torch nije instaliran; postavi CVOICE_TORCH_INDEX i probaj ponovo")
        if not args.no_asr and os.environ.get("CVOICE_WITH_ASR", "1") == "1":
            say(f"{D}· instaliram ASR proveru (opciono)…{X}")
            if run([vpy, "-m", "pip", "install", "-q", "-e", ".[asr]"], cwd=ROOT).returncode != 0:
                warn("ASR ekstra nije instaliran; radiće bez ocenjivanja")

    # ---- config ----
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("CVOICE_CONFIG_DIR", str(config_dir()))
    from cvoice import config as C  # noqa: E402

    cfg = C.load()
    interactive = sys.stdin.isatty()
    token = args.token or ""

    if role in ("server", "both"):
        token = token or cfg["server"].get("token") or secrets.token_urlsafe(32)
        host = args.bind or os.environ.get("CVOICE_BIND") or cfg["server"]["host"]
        port = args.port or int(os.environ.get("CVOICE_PORT") or cfg["server"]["port"])
        if interactive and not args.bind:
            hint = ("127.0.0.1 = samo ova mašina; unesi tailscale/LAN adresu "
                    "da bi i drugi uređaji mogli") if role == "both" else \
                   "127.0.0.1 = samo lokalno; 0.0.0.0 = mreža"
            h = input(f"\n{B}Na koju adresu da sluša?{X} {D}({hint}){X} [{host}] ").strip()
            host = h or host
            p = input(f"{B}Port{X} [{port}] ").strip()
            port = int(p) if p.isdigit() else port
        cfg["server"].update({"host": host, "port": port, "token": token})
        cfg["client"]["server"] = "http://%s:%d" % (host.replace("0.0.0.0", "127.0.0.1"), port)
        cfg["client"]["token"] = token

    if role == "client":
        url = args.server_url or os.environ.get("CVOICE_SERVER") or cfg["client"]["server"]
        if interactive and not args.server_url:
            u = input(f"\n{B}Na koji server da se povezuje?{X} [{url}] ").strip()
            url = u or url
            t = input(f"{B}Token servera{X} {D}(prazno ako server nema token){X}: ").strip()
            token = t or os.environ.get("CVOICE_TOKEN", "")
        else:
            token = token or os.environ.get("CVOICE_TOKEN", cfg["client"].get("token", ""))
        cfg["client"]["server"] = url
        if token:
            cfg["client"]["token"] = token

    path = C.save(cfg)
    ok(f"konfiguracija: {path}")

    # ---- launchers ----
    bd = bin_dir()
    bd.mkdir(parents=True, exist_ok=True)
    made = []
    for name in ("reci", "cvoice", "cvoiced"):
        src = venv_script(venv, name)
        if not src.exists() or (name == "cvoiced" and role == "client"):
            continue
        dst = bd / (name + (".exe" if WIN else ""))
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if WIN:
                # Symlinks need admin or developer mode on Windows; a .cmd shim
                # is boring and always works.
                shim = bd / (name + ".cmd")
                shim.write_text('@echo off\r\n"%s" %%*\r\n' % src, encoding="utf-8")
                made.append(shim.name)
                continue
            dst.symlink_to(src)
            made.append(name)
        except OSError as exc:
            warn(f"ne mogu da napravim {dst}: {exc}")
    ok("komande u %s: %s" % (bd, " ".join(made) or "(nijedna)"))
    if str(bd) not in os.environ.get("PATH", ""):
        warn(f"{bd} nije u PATH-u")

    # ---- service manager (server roles) ----
    if role != "client" and MAC:
        plist_src = ROOT / "launchd/com.bogdan.cvoiced.plist"
        if plist_src.exists():
            agents = Path.home() / "Library/LaunchAgents"
            agents.mkdir(parents=True, exist_ok=True)
            logs = Path.home() / "Library/Logs"
            logs.mkdir(parents=True, exist_ok=True)
            dst = agents / "com.bogdan.cvoiced.plist"
            dst.write_text(
                plist_src.read_text(encoding="utf-8")
                .replace("@EXEC@", str(venv_script(venv, "cvoiced")))
                .replace("@LOG@", str(logs)), encoding="utf-8")
            ok(f"launchd agent: {dst}")
            say(f"{D}   pokreni sa: launchctl load -w {dst}{X}")
    if role != "client" and not WIN and not MAC:
        unit_src = ROOT / "systemd/cvoiced.service"
        if unit_src.exists():
            udir = Path.home() / ".config/systemd/user"
            udir.mkdir(parents=True, exist_ok=True)
            (udir / "cvoiced.service").write_text(
                unit_src.read_text(encoding="utf-8").replace(
                    "@EXEC@", str(venv_script(venv, "cvoiced"))), encoding="utf-8")
            run(["systemctl", "--user", "daemon-reload"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok(f"systemd jedinica: {udir / 'cvoiced.service'}")
            say(f"{D}   pokreni sa: systemctl --user enable --now cvoiced{X}")

    say()
    ok(f"gotovo ({role})")
    if role != "client":
        say()
        say(f"  {B}Token servera:{X} {token}")
        say(f"  {D}Na klijentu: ownbox install cvoice → uloga 1 → ovaj token{X}")
        say(f"  {D}Model (~3 GB) se skida pri prvom generisanju, ne sada.{X}")
        if MAC or WIN:
            say(f"  {Y}Server na {'macOS' if MAC else 'Windows'} nije testiran — "
                f"radiće, ali sporo bez CUDA.{X}")
    say()
    say(f"  {B}cvoice{X}          {D}napravi profil{X}")
    say(f"  {B}reci \"tekst\"{X}    {D}izgovori nešto{X}")
    say(f"  {B}reci -l{X}         {D}izlistaj profile{X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
