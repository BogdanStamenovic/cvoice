#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove what cvoice created: service unit, config, profiles, downloaded models.

Run by `ownbox uninstall cvoice` via the manifest's remove hook, or directly.

Two rules govern what this deletes:

  1. It only removes things cvoice created. A model path you pointed the server
     at yourself is yours, not ours, and is left alone - only weights cvoice
     itself pulled into the Hugging Face cache are removed.
  2. Voice profiles are archived before deletion. Models re-download in minutes;
     a voice profile means getting the person back in a quiet room. --purge
     skips the archive if you really mean it.

Always exits 0. ownbox runs remove hooks with check=True, so a non-zero exit
here would abort the uninstall and strand the checkout.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

WIN = platform.system() == "Windows"
MAC = platform.system() == "Darwin"
B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
if WIN and not os.environ.get("WT_SESSION"):
    B = D = G = Y = R = X = ""

removed: list[str] = []
kept: list[str] = []


def config_dir() -> Path:
    if os.environ.get("CVOICE_CONFIG_DIR"):
        return Path(os.environ["CVOICE_CONFIG_DIR"])
    if WIN:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "cvoice"
    return Path.home() / ".config/cvoice"


def load_cfg() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return {}
    p = config_dir() / "config.toml"
    if not p.exists():
        return {}
    try:
        with p.open("rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p, onerror=lambda e: None):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def rm(p: Path, label: str) -> None:
    try:
        if p.is_dir() and not p.is_symlink():
            size = dir_size(p)
            shutil.rmtree(p)
            removed.append(f"{label}  {human(size)}  {p}")
        elif p.exists() or p.is_symlink():
            size = p.stat().st_size if p.exists() else 0
            p.unlink()
            removed.append(f"{label}  {human(size)}  {p}")
    except OSError as exc:
        print(f"{Y}! ne mogu da obrišem {p}: {exc}{X}", file=sys.stderr)


def stop_service() -> None:
    if MAC:
        plist = Path.home() / "Library/LaunchAgents/io.cvoice.daemon.plist"
        if plist.exists():
            subprocess.run(["launchctl", "unload", "-w", str(plist)],
                           capture_output=True, timeout=60)
            rm(plist, "launchd agent")
        return
    if WIN:
        return
    unit = Path.home() / ".config/systemd/user/cvoiced.service"
    if unit.exists():
        for args in (["--user", "disable", "--now", "cvoiced"],
                     ["--user", "reset-failed", "cvoiced"]):
            subprocess.run(["systemctl", *args], capture_output=True, timeout=60)
        rm(unit, "systemd unit")
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                       capture_output=True, timeout=60)


def hf_cache_dirs(model_ids) -> list[Path]:
    """Cache directories for models cvoice pulled. A model_id that is a local
    path was supplied by the user and is deliberately not matched."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = home / "hub"
    out = []
    if not hub.is_dir():
        return out
    for mid in model_ids:
        if not mid or os.sep in str(mid) and Path(mid).exists():
            continue
        name = str(mid).strip("/")
        if "/" in name:
            cand = hub / ("models--" + name.replace("/", "--"))
        else:
            cand = hub / ("models--" + name)
        if cand.is_dir():
            out.append(cand)
        else:
            for d in hub.glob("models--*"):
                if d.name.lower().endswith(name.replace("/", "--").lower()):
                    out.append(d)
    return out


def archive_profiles(profiles: Path) -> Path | None:
    if not profiles.is_dir() or not any(profiles.iterdir()):
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path.home() / f"cvoice-profiles-{stamp}.tar.gz"
    try:
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(profiles, arcname="profiles")
        return dest
    except OSError as exc:
        print(f"{Y}! arhiviranje profila nije uspelo: {exc}{X}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove cvoice data")
    ap.add_argument("--purge", action="store_true",
                    help="ne arhiviraj profile pre brisanja")
    ap.add_argument("--keep-data", action="store_true",
                    help="ostavi profile, modele i konfiguraciju")
    ap.add_argument("--yes", "-y", action="store_true", help="ne pitaj")
    args = ap.parse_args()

    cfg = load_cfg()
    scfg = cfg.get("server", {}) or {}
    data_dir = Path(str(scfg.get("data_dir") or Path.home() / ".local/share/cvoice")).expanduser()
    profiles = data_dir / "profiles"
    models = hf_cache_dirs([scfg.get("model"), scfg.get("asr_model")])

    n_prof = len([d for d in profiles.iterdir() if d.is_dir()]) if profiles.is_dir() else 0
    model_bytes = sum(dir_size(m) for m in models)

    print(f"\n{B}cvoice — brisanje podataka{X}")
    print(f"  profili   : {n_prof}" + (f"  ({profiles})" if n_prof else ""))
    print(f"  modeli    : {len(models)}  {human(model_bytes)}")
    print(f"  konfig    : {config_dir() / 'config.toml'}")
    local_model = scfg.get("model")
    if local_model and os.sep in str(local_model) and Path(str(local_model)).exists():
        print(f"  {D}model na tvojoj putanji ostaje netaknut: {local_model}{X}")

    if args.keep_data:
        print(f"{Y}--keep-data: brišem samo servis{X}")
        stop_service()
    else:
        if not args.yes and sys.stdin.isatty() and n_prof:
            ans = input(f"\n{B}Obrisati {n_prof} glasovn(a/ih) profil(a)?{X} "
                        f"{D}(arhiva se pravi prvo){X} [D/n] ").strip().lower()
            if ans in ("n", "ne", "no"):
                args.keep_data = True

        stop_service()
        if not args.keep_data:
            if not args.purge:
                arch = archive_profiles(profiles)
                if arch:
                    print(f"{G}✓{X} profili arhivirani: {B}{arch}{X}")
                    kept.append(str(arch))
            for m in models:
                rm(m, "model")
            rm(data_dir, "podaci")
            rm(config_dir(), "konfiguracija")
        else:
            kept.append(str(data_dir))
            kept.append(str(config_dir()))

    print()
    if removed:
        print(f"{B}Obrisano:{X}")
        for r in removed:
            print(f"  {r}")
    if kept:
        print(f"{B}Zadržano:{X}")
        for k in kept:
            print(f"  {k}")
    if not removed and not kept:
        print(f"{D}  ništa za brisanje{X}")
    print()
    # Never fail: ownbox runs remove hooks with check=True and a non-zero exit
    # would abort the uninstall and leave the checkout behind.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"{Y}! cvoice uninstall: {exc}{X}", file=sys.stderr)
        raise SystemExit(0)
