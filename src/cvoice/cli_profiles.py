# -*- coding: utf-8 -*-
"""`cvoice profiles ...` - list, share, get, remove.

Sharing goes through fiotransfer, which uploads to anonymous temporary file
hosts and prints a compact code. Codes look like `t:AdOxr/part` - provider
prefix, colon, then a path containing slashes - so they are always passed to
fioget as a single argv element rather than interpolated into a shell string.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import config
from .client import Client, ServerError
from .profiles import slugify

B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def work_dir() -> Path:
    # Not /tmp: it is tmpfs on many Linux desktops, so a profile archive there
    # is spent RAM rather than disk.
    d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cvoice"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        print(f"{R}nedostaje '{tool}'{X}", file=sys.stderr)
        print(f"{D}  instaliraj sa: ownbox install fiotransfer{X}", file=sys.stderr)
        raise SystemExit(1)
    return path


def cmd_list(cli: Client) -> int:
    try:
        profs = cli.profiles()
    except Exception as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    if not profs:
        print("(nema profila — napravi ga sa 'cvoice')"); return 0
    print(f"  {'SLUG':<26} {'IME':<28} {'TRAJANJE':>8}  NAPOMENA")
    for p in profs:
        print("  %-26s %-28s %7.1fs  %s"
              % (p["slug"], p.get("name", ""), p.get("duration") or 0, p.get("notes", "")))
    return 0


def cmd_share(cli: Client, name: str, keep: bool) -> int:
    _need("fiotransfer")
    try:
        blob = cli.export_profile(name)
    except ServerError as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    except Exception as exc:
        print(f"{R}server nedostupan: {exc}{X}", file=sys.stderr); return 1

    slug = slugify(name)
    archive = work_dir() / f"cvoice-{slug}.tar.gz"
    archive.write_bytes(blob)
    size = archive.stat().st_size
    print(f"{D}  {archive.name} · {size / 1024:.0f} KB{X}")
    print(f"{Y}  napomena: fiotransfer šalje na javne anonimne servere — "
          f"ko ima kod, ima i glas.{X}")

    proc = subprocess.run(["fiotransfer", str(archive)], capture_output=True,
                          text=True, timeout=1800)
    out = (proc.stdout or "") + (proc.stderr or "")
    if not keep:
        archive.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"{R}upload nije uspeo{X}", file=sys.stderr)
        print(out.strip()[-600:], file=sys.stderr)
        return 1
    m = re.search(r"^Code:\s*(\S+)\s*$", out, re.M)
    if not m:
        print(f"{R}ne mogu da pročitam kod iz izlaza fiotransfer-a{X}", file=sys.stderr)
        print(out.strip()[-600:], file=sys.stderr)
        return 1
    code = m.group(1)
    print()
    print(f"{G}{B}  {code}{X}")
    print(f"{D}  drugi to preuzima sa:{X}  {B}cvoice profiles get {code}{X}")
    return 0


def cmd_get(cli: Client, code: str, new_name: str | None, save_only: bool) -> int:
    _need("fioget")
    dest = work_dir() / "incoming.tar.gz"
    dest.unlink(missing_ok=True)
    print(f"{D}· preuzimam {code}…{X}")
    proc = subprocess.run(["fioget", code, str(dest)], capture_output=True,
                          text=True, timeout=1800)
    if proc.returncode != 0 or not dest.exists():
        print(f"{R}preuzimanje nije uspelo{X}", file=sys.stderr)
        print(((proc.stdout or "") + (proc.stderr or "")).strip()[-600:], file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(dir=work_dir()) as td:
        try:
            with tarfile.open(dest, "r:gz") as tar:
                # Never let an archive write outside the extraction directory.
                root = Path(td).resolve()
                for member in tar.getmembers():
                    target = (root / member.name).resolve()
                    if not str(target).startswith(str(root)):
                        print(f"{R}arhiva sadrži putanju van direktorijuma: "
                              f"{member.name}{X}", file=sys.stderr)
                        return 1
                    if member.issym() or member.islnk():
                        print(f"{R}arhiva sadrži link: {member.name}{X}", file=sys.stderr)
                        return 1
                tar.extractall(td)
        except tarfile.TarError as exc:
            print(f"{R}nije validna cvoice arhiva: {exc}{X}", file=sys.stderr); return 1

        wavs = list(Path(td).rglob("ref.wav"))
        if not wavs:
            print(f"{R}u arhivi nema ref.wav — nije cvoice profil{X}", file=sys.stderr)
            return 1
        pdir = wavs[0].parent
        text = ""
        if (pdir / "ref.txt").exists():
            text = (pdir / "ref.txt").read_text(encoding="utf-8").strip()
        name = new_name
        if not name and (pdir / "meta.json").exists():
            import json
            try:
                name = json.loads((pdir / "meta.json").read_text(encoding="utf-8")).get("name")
            except Exception:
                name = None
        name = name or pdir.name

        if save_only:
            out = work_dir() / f"cvoice-{slugify(name)}.tar.gz"
            shutil.move(str(dest), out)
            print(f"{G}✓{X} sačuvano: {B}{out}{X}  {D}(nije instalirano){X}")
            return 0

        existing = {p["slug"] for p in cli.profiles()}
        if slugify(name) in existing:
            ans = input(f"{Y}Profil '{slugify(name)}' već postoji. Prepisati? [d/N] {X}") \
                if sys.stdin.isatty() else "n"
            if ans.strip().lower() not in ("d", "da", "y", "yes"):
                print("prekinuto."); return 0
        try:
            res = cli.enrol(name, wavs[0], text, notes="shared")
        except ServerError as exc:
            print(f"{R}{exc}{X}", file=sys.stderr); return 1
    dest.unlink(missing_ok=True)
    meta = res.get("profile", {})
    print(f"\n{G}{B}Instalirano.{X}  {B}{meta.get('name', name)}{X}  "
          f"{D}({meta.get('slug')}, {meta.get('duration')}s){X}\n")
    print(f"  {B}reci -p {meta.get('slug')} \"tekst\"{X}")
    return 0


def cmd_rm(cli: Client, name: str) -> int:
    try:
        profs = {p["slug"] for p in cli.profiles()}
    except Exception as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    slug = name if name in profs else slugify(name)
    if slug not in profs:
        print(f"{R}nema profila: {name}{X}", file=sys.stderr)
        print(f"{D}  imaš: {', '.join(sorted(profs)) or '(nijedan)'}{X}", file=sys.stderr)
        return 1
    if sys.stdin.isatty():
        if input(f"{Y}Obrisati '{slug}'? [d/N] {X}").strip().lower() not in ("d", "da", "y"):
            print("prekinuto."); return 0
    try:
        cli.delete(slug)
    except Exception as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    print(f"{G}✓{X} obrisan {slug}")
    return 0


def main(argv) -> int:
    import argparse

    cfg = config.load()
    ap = argparse.ArgumentParser(
        prog="cvoice profiles", description="Upravljanje glasovnim profilima")
    ap.add_argument("--server", default=cfg["client"]["server"])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="izlistaj profile")

    sh = sub.add_parser("share", help="podeli profil preko fiotransfer-a")
    sh.add_argument("name")
    sh.add_argument("--keep", action="store_true", help="ostavi .tar.gz posle uploada")

    gt = sub.add_parser("get", help="preuzmi profil po kodu")
    gt.add_argument("code")
    gt.add_argument("--name", help="instaliraj pod drugim imenom")
    gt.add_argument("--save-only", action="store_true",
                    help="samo preuzmi arhivu, ne instaliraj")

    rm = sub.add_parser("rm", help="obriši profil")
    rm.add_argument("name")

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help(); return 0

    cli = Client(args.server, cfg["client"].get("token", ""))
    if args.cmd == "list":
        return cmd_list(cli)
    if args.cmd == "share":
        return cmd_share(cli, args.name, args.keep)
    if args.cmd == "get":
        return cmd_get(cli, args.code, args.name, args.save_only)
    if args.cmd == "rm":
        return cmd_rm(cli, args.name)
    return 0
