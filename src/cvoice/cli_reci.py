# -*- coding: utf-8 -*-
"""reci - say a line in a cloned voice."""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from . import config, discord
from .client import Client, ServerError

B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"

HINT = f"""{B}Saveti za tekst{X}
  {B}dijakritika je obavezna{X} — "šifra", ne "sifra"
        (bez nje greška u izgovoru raste otprilike četvorostruko)
  {B},{X}      pauza za dah        {B}...{X}   oklevanje
  {B}.{X}      zatvara misao       {B}aaa{X}   razvučeno dozivanje (~+9% trajanja)"""


def _looks_undiacriticked(text: str) -> bool:
    if any(c in text for c in "čćžšđČĆŽŠĐ"):
        return False
    return bool(any(c.isalpha() for c in text)) and len(text.split()) >= 2


def main() -> int:
    cfg = config.load()
    ap = argparse.ArgumentParser(
        prog="reci", description="Izgovori tekst kloniranim glasom.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HINT)
    ap.add_argument("text", nargs="*", help="šta da kaže (bez ovoga: pita te)")
    ap.add_argument("-p", "--profile", default=cfg["client"].get("profile", ""),
                    help="čiji glas (slug, puno ime ili prefiks)")
    ap.add_argument("-t", "--takes", type=int, default=int(cfg["client"].get("takes", 3)),
                    help="koliko pokušaja pa se bira najbolji (default 3)")
    ap.add_argument("-1", dest="one", action="store_true", help="jedan pokušaj, bez provere")
    ap.add_argument("-s", "--save", help="snimi .wav i ovde")
    ap.add_argument("-d", "--dry", action="store_true", help="ne šalji na Discord")
    ap.add_argument("-l", "--list", action="store_true", help="izlistaj profile i izađi")
    ap.add_argument("--server", default=cfg["client"]["server"])
    ap.add_argument("--lang", default=None)
    args = ap.parse_args()

    cli = Client(args.server, cfg["client"].get("token", ""))

    if args.list:
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

    text = " ".join(args.text).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print(f"{B}Šta da kaže?{X}")
        print(f"{D}(dijakritika obavezna · , = dah · ... = oklevanje · aaa = razvučeno){X}")
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 1
    if not text:
        print(f"{R}prazan tekst{X}", file=sys.stderr); return 1

    if _looks_undiacriticked(text):
        print(f"{Y}! nema dijakritike — proveri da ne treba č/ć/ž/š/đ{X}", file=sys.stderr)

    takes = 1 if args.one else max(1, args.takes)
    print(f"{D}  {text}{X}")
    print(f"{D}· glas: {args.profile or '(default sa servera)'} · {takes} pokušaj(a){X}")
    print(f"{D}· {args.server}{X}")

    try:
        res = cli.speak(text, profile=args.profile, takes=takes, language=args.lang)
    except ServerError as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    except Exception as exc:
        print(f"{R}server nedostupan: {exc}{X}", file=sys.stderr); return 1

    if res.get("scores"):
        print()
        print(f"  {'WER':<7}{'CER':<7}šta se čuje")
        for s in res["scores"][:5]:
            col = G if s["wer"] <= 10 else (Y if s["wer"] <= 25 else R)
            print(f"  {col}{s['wer']:<6.1f}{X} {s['cer']:<6.1f} {s['transcript'][:58]}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.save) if args.save else Path(tempfile.gettempdir()) / f"reci-{stamp}.wav"
    out.write_bytes(res["audio"])
    print()
    if args.save:
        print(f"{G} snimljeno: {out}{X}")

    if args.dry or not discord.configured(cfg):
        if not args.dry and not discord.configured(cfg):
            print(f"{D}· Discord nije podešen — fajl: {out}{X}")
        else:
            print(f"{Y}· dry run — nije poslato. Fajl: {out}{X}")
        return 0

    try:
        mid = discord.post(cfg, out, content=f"🔊 **{res['profile']['name']}** — {text[:90]}",
                           filename=f"reci-{stamp}.wav")
        print(f"{G}✓ poslato na Discord{X} {D}({mid}){X}")
    except Exception as exc:
        print(f"{R}✗ Discord: {exc}{X}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
