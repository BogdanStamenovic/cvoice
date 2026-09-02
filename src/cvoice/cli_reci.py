# -*- coding: utf-8 -*-
"""reci - say a line in a cloned voice."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
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


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class _Progress(threading.Thread):
    """Show what the server is doing while /speak blocks.

    The first call after a fresh install downloads ~3 GB of weights inside the
    server process. Without this the client just sits there, and a silent
    multi-minute wait is indistinguishable from a hang.
    """

    def __init__(self, client, delay=2.0):
        super().__init__(daemon=True)
        self.client, self.delay = client, delay
        self.stop = threading.Event()
        self.printed = False

    def run(self):
        if self.stop.wait(self.delay):
            return
        while not self.stop.is_set():
            try:
                st = self.client.status()
            except Exception:
                st = {}
            phase = st.get("phase", "")
            if phase == "downloading":
                got, total = st.get("downloaded") or 0, st.get("total")
                rate = st.get("rate") or 0
                bar = ""
                if total:
                    frac = min(got / total, 1.0)
                    filled = int(frac * 24)
                    bar = " [%s%s] %3.0f%%" % ("█" * filled, "·" * (24 - filled), frac * 100)
                    eta = (total - got) / rate if rate > 0 else 0
                    tail = f"  ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta > 1 else ""
                else:
                    tail = ""
                sys.stderr.write(
                    f"\r{D}  skidam model{bar} {_human(got)}"
                    + (f" / {_human(total)}" if total else "")
                    + (f" · {_human(rate)}/s" if rate > 0 else "") + tail + f"{X}   ")
                sys.stderr.flush()
                self.printed = True
            elif phase == "loading":
                sys.stderr.write(f"\r{D}  učitavam model…{X}                      ")
                sys.stderr.flush()
                self.printed = True
            self.stop.wait(1.0)

    def done(self):
        self.stop.set()
        if self.printed:
            sys.stderr.write("\r" + " " * 78 + "\r")
            sys.stderr.flush()


def _doctor(cli, cfg) -> int:
    """Check everything an install needs, without recording or generating."""
    import platform as _plat
    from . import audio

    ok_mark, bad, warn = f"{G}✓{X}", f"{R}✗{X}", f"{Y}!{X}"
    rc = 0
    print(f"{B}cvoice doctor{X}")
    print(f"  platforma   : {_plat.system()} {_plat.release()} · {_plat.machine()}")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  konfig      : {config.CONFIG_PATH}"
          f"{'' if config.CONFIG_PATH.exists() else f'  {warn} ne postoji'}")

    print(f"\n{B}zvuk (potreban samo za cvoice, ne za reci){X}")
    if audio.have_capture():
        devs = audio.sources()
        print(f"  {ok_mark} PortAudio radi · {len(devs)} ulaznih uređaja")
        for d in devs[:4]:
            tag = audio.describe(d)
            print(f"      {d['index']}: {d['name'][:48]}" + (f"  ({tag})" if tag else ""))
        if not devs:
            print(f"      {warn} nijedan mikrofon — {audio.silence_hint()}")
    else:
        print(f"  {bad} sounddevice/PortAudio nedostupan"); rc = 1
    print(f"  pojačanje   : {'automatsko' if audio.gain_controllable() else 'ručno — ' + audio.manual_gain_hint()}")

    print(f"\n{B}server{X}")
    print(f"  adresa      : {cli.base}")
    try:
        h = cli.health()
        print(f"  {ok_mark} dostupan · v{h.get('version')} · jezik {h.get('language')}")
        print(f"      model učitan : {h.get('model_loaded')}")
        print(f"      ASR dostupan : {h.get('asr_available')}")
        print(f"      profila      : {h.get('profiles')}")
        try:
            profs = cli.profiles()
            print(f"  {ok_mark} token prihvaćen · {len(profs)} profil(a)")
        except Exception:
            print(f"  {bad} token odbijen — proveri client.token"); rc = 1
    except Exception as exc:
        print(f"  {bad} nedostupan: {exc}")
        print(f"      {D}pokreni server, ili promeni client.server u konfiguraciji{X}")
        rc = 1
    return rc


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
    ap.add_argument("--doctor", action="store_true",
                    help="proveri instalaciju (platforma, zvuk, server) i izađi")
    ap.add_argument("--server", default=cfg["client"]["server"])
    ap.add_argument("--lang", default=None)
    args = ap.parse_args()

    cli = Client(args.server, cfg["client"].get("token", ""))

    if args.doctor:
        return _doctor(cli, cfg)

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

    prog = _Progress(cli)
    prog.start()
    try:
        res = cli.speak(text, profile=args.profile, takes=takes, language=args.lang)
    except ServerError as exc:
        prog.done()
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    except Exception as exc:
        prog.done()
        print(f"{R}server nedostupan: {exc}{X}", file=sys.stderr); return 1
    finally:
        prog.done()

    if res.get("scores"):
        print()
        print(f"  {'WER':<7}{'CER':<7}šta se čuje")
        for s in res["scores"][:5]:
            col = G if s["wer"] <= 10 else (Y if s["wer"] <= 25 else R)
            print(f"  {col}{s['wer']:<6.1f}{X} {s['cer']:<6.1f} {s['transcript'][:58]}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.save:
        out = Path(args.save)
    else:
        # NOT tempfile.gettempdir(): /tmp is tmpfs on many Linux desktops
        # (sized at half of RAM), so writing audio there spends memory and
        # never reclaims it. Use the on-disk cache directory instead.
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cvoice"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"reci-{stamp}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
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
