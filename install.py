#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cvoice installer. Runs on Linux, macOS and Windows.

Python rather than shell because Windows has no bash. Invoked directly or by
`ownbox install cvoice`; ownbox runs setup with stdin/stdout attached, so the
role prompt works. Non-interactive callers pass --role or set CVOICE_ROLE; with
neither, a box that already has a server token is taken to be a server.

`--update` is what `ownbox update` runs: no prompts, the role and venv from the
last install (config_dir/install.json), the existing config left untouched.

The venv can live outside the checkout (--venv, CVOICE_VENV, or the shared
default below), and is then linked in as .venv. That is not a nicety: a server
venv is ~7.5 GB (torch 1.2 GB, nvidia-* 4.4 GB, measured 2026-11-02) and this
box's root has a few GB free, so a second copy per checkout does not fit. An
existing venv is reused and repaired in place, never rebuilt beside itself.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = platform.system() == "Windows"
MAC = platform.system() == "Darwin"

# Where archserver keeps the one server venv: an ext4 image on the big Windows
# disk, because root cannot hold it and NTFS cannot hold a venv (no symlinks).
# Used only when it already exists; elsewhere the venv goes in the checkout.
SHARED_VENV = Path("/mnt/offload/cvoice-venv")
# A server venv measured 7.5 GB; refuse to start building one with less free.
SERVER_VENV_BYTES = 8 * 2**30

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


def state_path() -> Path:
    return config_dir() / "install.json"


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def inferred_role() -> str:
    """Without a TTY nobody can answer the prompt. A config that already holds
    a server token belongs to a server; defaulting to "client" there (the old
    behaviour) would reinstall a live server as a client under `--yes`."""
    p = config_dir() / "config.toml"
    try:
        import tomllib
        with p.open("rb") as fh:
            if (tomllib.load(fh).get("server") or {}).get("token"):
                return "both"
    except Exception:
        pass
    return "client"


def venv_ok(venv: Path, need_torch: bool) -> bool:
    probe = "import torch" if need_torch else "import sys"
    try:
        return run([str(venv_python(venv)), "-c", probe], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pick_venv(args, st: dict, role: str) -> Path:
    """The venv to use, in order: --venv, CVOICE_VENV, the last install's,
    whatever .venv in this checkout already points at, the shared server venv
    if it exists, and only then a fresh .venv here."""
    for cand in (args.venv, os.environ.get("CVOICE_VENV"), st.get("venv")):
        if cand:
            return Path(cand).expanduser()
    local = ROOT / ".venv"
    if local.exists():
        return local.resolve()
    if role != "client" and not WIN and not MAC and SHARED_VENV.exists():
        return SHARED_VENV
    return local


def link_venv(venv: Path) -> None:
    """Make ROOT/.venv lead to `venv`, so manifest commands and the unit can say
    .venv/bin/... wherever the venv really is."""
    local = ROOT / ".venv"
    if venv == local or (local.exists() and local.resolve() == venv.resolve()):
        return
    if local.exists() and not local.is_symlink():
        die(f"{local} je pravi venv, a tražen je {venv}; obriši jedan od njih")
    if local.is_symlink():
        local.unlink()
    local.symlink_to(venv, target_is_directory=True)
    ok(f".venv → {venv}")


def editable_source(vpy: str) -> str:
    """The checkout the venv's editable cvoice currently points at, or ""."""
    code = ("import json,importlib.metadata as m\n"
            "try: print(json.loads(m.distribution('cvoice').read_text('direct_url.json'))['url'])\n"
            "except Exception: pass")
    r = run([vpy, "-c", code], capture_output=True, text=True)
    url = (r.stdout or "").strip()
    return url[len("file://"):] if url.startswith("file://") else ""


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def prompt_role(default="client") -> str:
    if not sys.stdin.isatty():
        role = os.environ.get("CVOICE_ROLE") or inferred_role() or default
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
    ap.add_argument("--venv", help="venv to use or create (default: reuse an existing one)")
    ap.add_argument("--show-token", action="store_true",
                    help="print the server token in full at the end")
    ap.add_argument("--update", action="store_true",
                    help="upgrade in place: no prompts, keep config, restart the daemon")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        die("treba Python 3.10+")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    st = load_state()
    if args.update:
        role = args.role or st.get("role") or os.environ.get("CVOICE_ROLE") or inferred_role()
    else:
        role = args.role or os.environ.get("CVOICE_ROLE") or prompt_role()
    say(f"{D}· uloga: {role}{X}")

    venv = pick_venv(args, st, role)
    if not venv_python(venv).exists():
        need = SERVER_VENV_BYTES if role != "client" else 2**30
        if free_bytes(venv) < need:
            die(f"nema mesta za venv u {venv}: slobodno {free_bytes(venv) / 2**30:.1f} GB, "
                f"treba ~{need / 2**30:.0f} GB. Pokaži na postojeći sa --venv / CVOICE_VENV.")
        say(f"{D}· pravim venv u {venv}…{X}")
        if run([sys.executable, "-m", "venv", str(venv)]).returncode != 0:
            die("ne mogu da napravim venv")
        fresh = True
    else:
        fresh = False
        if not venv_ok(venv, need_torch=False):
            die(f"venv {venv} postoji ali ne radi; popravi ga ili obriši")
        ok(f"koristim postojeći venv: {venv}")
    link_venv(venv)
    # The venv's real python, not ROOT/.venv/...: pip writes sys.executable into
    # every console script's shebang, and a shebang through one checkout's
    # .venv link breaks every other user of the venv when that checkout goes.
    vpy = str(venv_python(venv.resolve()))
    if fresh:
        run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Remember who the editable install pointed at before we take it over, so
    # uninstall.py can hand a shared venv back to the dev checkout.
    prev = editable_source(vpy)
    if prev and Path(prev).resolve() != ROOT and not st.get("previous_editable"):
        st["previous_editable"] = prev

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
    interactive = sys.stdin.isatty() and not args.update
    token = args.token or ""
    # An existing config is the user's: kept on --update, and on any run with
    # nobody to ask and nothing given on the command line (ownbox --yes on a box
    # that already has cvoice configured, as when moving it onto ownbox).
    explicit = any((args.server_url, args.token, args.bind, args.port))
    fresh_token = False
    keep_config = C.CONFIG_PATH.exists() and (
        args.update or (not sys.stdin.isatty() and not explicit))

    if role in ("server", "both") and not keep_config:
        fresh_token = not (token or cfg["server"].get("token"))
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

    if role == "client" and not keep_config:
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

    if keep_config:
        # An upgrade never rewrites the user's config: save() only knows three
        # sections and would drop anything else they put there.
        token = cfg["server"].get("token") or cfg["client"].get("token") or ""
        ok(f"konfiguracija: {C.CONFIG_PATH} (netaknuta)")
    else:
        path = C.save(cfg)
        ok(f"konfiguracija: {path}")

    # ---- launchers ----
    # Under ownbox, ownbox owns the launchers. Creating our own here leaves a
    # file it did not generate and will refuse to replace, which breaks every
    # later `ownbox update`. Detect its checkout layout and stay out of the way.
    managed = "ownbox" in str(ROOT).split(os.sep)
    if managed:
        ok("komande: ownbox ih pravi (preskačem)")
    bd = bin_dir()
    bd.mkdir(parents=True, exist_ok=True)
    made = []
    for name in ([] if managed else ("cvoice", "cvoiced")):
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
    if not managed:
        ok("komande u %s: %s" % (bd, " ".join(made) or "(nijedna)"))
        if str(bd) not in os.environ.get("PATH", ""):
            warn(f"{bd} nije u PATH-u")

    # ---- service manager (server roles) ----
    if role != "client" and MAC:
        plist_src = ROOT / "launchd/io.cvoice.daemon.plist"
        if plist_src.exists():
            agents = Path.home() / "Library/LaunchAgents"
            agents.mkdir(parents=True, exist_ok=True)
            logs = Path.home() / "Library/Logs"
            logs.mkdir(parents=True, exist_ok=True)
            dst = agents / "io.cvoice.daemon.plist"
            dst.write_text(
                plist_src.read_text(encoding="utf-8")
                .replace("@EXEC@", str(venv_script(venv, "cvoiced")))
                .replace("@LOG@", str(logs)), encoding="utf-8")
            ok(f"launchd agent: {dst}")
            say(f"{D}   pokreni sa: launchctl load -w {dst}{X}")
    if role != "client" and not WIN and not MAC:
        install_unit(venv, cfg)

    st.update(role=role, venv=str(venv), root=str(ROOT))
    save_state(st)

    say()
    ok(f"gotovo ({role})")
    if role != "client":
        say()
        # The token is a credential, and this output lands in ownbox logs and
        # agent transcripts on every install and update. Say where it lives;
        # show it only when asked, or on a terminal the one time it is new and
        # someone is about to type it into a client.
        if token and (args.show_token or (fresh_token and sys.stdin.isatty())):
            say(f"  {B}Token servera:{X} {token}")
        else:
            say(f"  {B}Token servera:{X} u {C.CONFIG_PATH} [server] token "
                f"{D}(pun prikaz: install.py --show-token){X}")
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


def mount_of(path: Path) -> Path:
    p = path.resolve()
    while not os.path.ismount(p):
        p = p.parent
    return p


def install_unit(venv: Path, cfg: dict) -> None:
    unit_src = ROOT / "systemd/cvoiced.service"
    if not unit_src.exists():
        return
    udir = Path.home() / ".config/systemd/user"
    udir.mkdir(parents=True, exist_ok=True)
    # ExecStart goes through this checkout's .venv link, so the unit names the
    # checkout it runs, whichever disk the venv itself is on.
    (udir / "cvoiced.service").write_text(
        unit_src.read_text(encoding="utf-8").replace(
            "@EXEC@", str(venv_script(ROOT / ".venv", "cvoiced"))), encoding="utf-8")
    ok(f"systemd jedinica: {udir / 'cvoiced.service'}")

    ddir = udir / "cvoiced.service.d"
    drop = ddir / "10-venv-mount.conf"
    mount = mount_of(venv)
    if mount != Path("/"):
        r = run(["systemd-escape", "--path", "--suffix=mount", str(mount)],
                capture_output=True, text=True)
        unit = (r.stdout or "").strip() or "%s.mount" % str(mount).strip("/").replace("/", "-")
        ddir.mkdir(parents=True, exist_ok=True)
        drop.write_text((ROOT / "systemd/10-venv-mount.conf").read_text(encoding="utf-8")
                        .replace("@MOUNT@", str(mount)).replace("@MOUNTUNIT@", unit),
                        encoding="utf-8")
        ok(f"systemd drop-in: {drop} ({mount})")
    elif drop.exists():
        drop.unlink()
    # The hand-written predecessor of 10-venv-mount.conf. Only ours is removed.
    legacy = ddir / "10-offload-mount.conf"
    try:
        if "the venv lives there" in legacy.read_text(encoding="utf-8"):
            legacy.unlink()
            ok(f"uklonjen stari drop-in: {legacy}")
    except OSError:
        pass

    run(["systemctl", "--user", "daemon-reload"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["systemctl", "--user", "enable", "cvoiced"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    restart_daemon(cfg)


def _main_pid() -> str:
    r = run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "cvoiced"],
            capture_output=True, text=True)
    return (r.stdout or "").strip()


def _health(base: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def restart_daemon(cfg: dict) -> None:
    """Bring cvoiced up on the new code without cutting anyone off.

    A running daemon is asked to POST /yield, which waits for its open requests
    and websockets (a phone call holds one) and then exits for systemd to
    restart. Busy past every retry: the old process is left running and we say
    so - an upgrade is never worth a dropped call. Only a daemon too old to have
    /yield, or one not answering at all, is restarted bluntly."""
    import urllib.error
    import urllib.request

    active = run(["systemctl", "--user", "is-active", "--quiet", "cvoiced"]).returncode == 0
    if not active:
        if run(["systemctl", "--user", "start", "cvoiced"]).returncode == 0:
            ok("cvoiced pokrenut")
        else:
            warn("cvoiced nije pokrenut; vidi: journalctl --user -u cvoiced")
        return
    srv = cfg.get("server", {})
    host = str(srv.get("host") or "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    base = "http://%s:%s" % (host, srv.get("port", 8760))
    token = srv.get("token") or ""
    before = _main_pid()
    blunt = False
    for attempt in range(1, 11):
        req = urllib.request.Request(base + "/yield?wait=30", method="POST", data=b"",
                                     headers={"Authorization": "Bearer " + token})
        try:
            with urllib.request.urlopen(req, timeout=60):
                break
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                warn(f"cvoiced je zauzet (pokušaj {attempt}/10), čekam…")
                continue
            blunt = True  # 404: a daemon from before /yield existed
            break
        except Exception:
            blunt = True
            break
    else:
        warn("cvoiced je i dalje zauzet; radi stari kod dok se ne restartuje: "
             "systemctl --user restart cvoiced")
        return
    if blunt:
        run(["systemctl", "--user", "restart", "cvoiced"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        pid = _main_pid()
        if pid not in ("", "0", before) and _health(base):
            ok(f"cvoiced restartovan ({'restart' if blunt else '/yield'}), pid {pid}")
            return
        time.sleep(1)
    warn("cvoiced se nije vratio za 90 s; vidi: journalctl --user -u cvoiced")


if __name__ == "__main__":
    raise SystemExit(main())
