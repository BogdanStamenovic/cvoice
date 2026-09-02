# -*- coding: utf-8 -*-
"""cvoice - enrol a new voice: calibrate, record ~20 s, test it, save it.

Recording happens locally because that is where the microphone is; cloning
happens on the server because that is where the GPU is.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import audio, config
from .client import Client, ServerError

B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
MIN_SECONDS = 12.0


def ask(prompt: str, default: str = "") -> str:
    try:
        v = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(1)
    return v or default


def pick_source() -> int | None:
    devs = audio.sources()
    if not devs:
        print(f"{R}Nema ulaznih uređaja.{X}"); sys.exit(1)
    print(f"{B}Mikrofoni:{X}")
    default = 1
    for i, d in enumerate(devs, 1):
        note = audio.describe(d)
        bt = audio.is_bluetooth(d["name"])
        colour = R if bt else G
        if not bt and (default == 1 or d["default"]):
            default = i
        star = " *" if d["default"] else ""
        print(f"  {i}) {d['name']}{star}" + (f"  {colour}({note}){X}" if note else ""))
    choice = ask(f"\nKoji? [{default}] ", str(default))
    try:
        dev = devs[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"{R}nevažeći izbor{X}"); sys.exit(1)
    if audio.is_bluetooth(dev["name"]):
        print(f"{R}Bluetooth mikrofoni (HFP/mSBC) seku visoke frekvencije koje su\n"
              f"enkoderu govornika potrebne. Izaberi drugi uređaj.{X}")
        sys.exit(1)
    return dev["index"]


def calibrate(dev: int | None, tmp: Path) -> None:
    """Close the loop on input level where the OS lets us, and guide the user
    where it does not (only Linux/pactl exposes a portable gain control)."""
    auto = audio.gain_controllable()
    print(f"\n{B}Kalibracija.{X} " +
          ("Ja nameštam nivo — ti samo pričaj normalno," if auto
           else "Proveravam nivo — pričaj normalno,"))
    print("uključujući i najglasniji deo.")
    if not auto:
        print(f"{D}({audio.manual_gain_hint()}){X}")
    cal = tmp / "cal.wav"
    for attempt in range(1, 9):
        ask(f"\n  [{attempt}] ENTER pa pričaj 5 sekundi. ")
        try:
            audio.record(cal, dev, seconds=5)
        except Exception as exc:
            print(f"  {R}snimanje nije uspelo: {exc}{X}"); return
        st = audio.stats(cal)
        cur = audio.gain_db()
        if st["duration"] < 0.5 or st["peak_dbfs"] < -60:
            print(f"  {R}ništa nije snimljeno — tišina.{X}")
            print(f"    {D}{audio.silence_hint()}{X}")
            continue
        print(f"      vrh {st['peak_dbfs']:.1f} dBFS" +
              (f" pri {cur:.1f} dB" if cur is not None else ""), end="")
        if audio.ACCEPT_LO <= st["peak_dbfs"] <= audio.ACCEPT_HI:
            print(f"  {G}→ dobro{X}")
            return
        too_hot = st["peak_dbfs"] > audio.ACCEPT_HI
        if auto and cur is not None:
            new = min(0.0, max(-60.0, cur + (audio.TARGET_PEAK_DBFS - st["peak_dbfs"])))
            print(f"  {Y}→ spuštam na {new:.1f} dB{X}")
            audio.set_gain_db(new)
        else:
            print(f"  {Y}→ {'preglasno, smanji' if too_hot else 'pretiho, pojačaj'} "
                  f"ulaz i probaj opet{X}")
            print(f"    {D}{audio.manual_gain_hint()}{X}")
    print(f"  {R}ne mogu da namestim; nastavljam{X}")


def record_passage(dev: int | None, tmp: Path, passage: str) -> Path:
    ref = tmp / "ref.wav"
    while True:
        print("\033[2J\033[H", end="")
        print(f"{B}Pročitaj ovo — oko 20 sekundi.{X}")
        print(f"{D}Normalnim glasom, ne kao spiker. Ravnomerno. "
              f"Ne pomeraj se od mikrofona.{X}\n")
        print(passage, "\n")
        ask("ENTER za snimanje, pa ENTER za kraj. ")
        audio.record_until_enter(ref, dev, "")
        st = audio.stats(ref)
        print(f"\n   {st['duration']:.1f}s · vrh {st['peak_dbfs']:.1f} dB "
              f"· klipovano {st['clipped_pct']:.3f}%  ", end="")
        if st["clipped_pct"] > audio.CLIP_TOLERANCE_PCT:
            print(f"{R}KLIPUJE — ponavljam{X}")
            cur = audio.gain_db()
            if audio.gain_controllable() and cur is not None:
                audio.set_gain_db(cur - 4)
                print(f"   {Y}spuštam pojačanje na {cur - 4:.1f} dB{X}")
            else:
                print(f"   {Y}smanji ulazno pojačanje: {audio.manual_gain_hint()}{X}")
            ask("   ENTER za novi pokušaj. ")
            continue
        if st["duration"] < MIN_SECONDS:
            print(f"{Y}kratko — treba bar {MIN_SECONDS:.0f} s{X}")
            ask("   ENTER za novi pokušaj. ")
            continue
        print(f"{G}dobro{X}")
        k = ask("   ENTER = zadrži, p = preslušaj, r = ponovi: ").lower()
        if k == "p":
            audio.play(ref)
            if ask("   ENTER = zadrži, r = ponovi: ").lower() == "r":
                continue
        elif k == "r":
            continue
        return ref


def main() -> int:
    # `cvoice` with no subcommand still enrols, which is the common case;
    # `cvoice profiles ...` routes to the management commands.
    if len(sys.argv) > 1 and sys.argv[1] == "profiles":
        from .cli_profiles import main as profiles_main
        return profiles_main(sys.argv[2:])

    cfg = config.load()
    ap = argparse.ArgumentParser(
        prog="cvoice", description="Napravi novi glasovni profil.",
        epilog="Takođe: cvoice profiles list|share|get|rm",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=cfg["client"]["server"])
    ap.add_argument("--name", help="preskoči pitanje za ime")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--from-wav", help="koristi postojeći snimak umesto mikrofona")
    ap.add_argument("--text", help="transkript uz --from-wav")
    args = ap.parse_args()

    cli = Client(args.server, cfg["client"].get("token", ""))
    try:
        h = cli.health()
    except Exception as exc:
        print(f"{R}server nedostupan ({args.server}): {exc}{X}", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="cvoice-"))
    passage = cli.passage(args.lang or h.get("language", "sr"))

    if args.from_wav:
        ref = Path(args.from_wav)
        passage = args.text or passage
        src = None
    else:
        if not audio.have_capture():
            print(f"{R}nedostaje sounddevice (PortAudio) — snimanje nije moguće{X}",
                  file=sys.stderr)
            print(f"{D}pip install sounddevice, ili koristi --from-wav{X}", file=sys.stderr)
            return 1
        print("\033[2J\033[H", end="")
        print(f"{B}cvoice{X} — novi glasovni profil\n")
        print("  Snimiš ~20 s, server klonira, ti kažeš da li valja, pa se snima pod imenom.")
        print(f"  Posle toga: {B}reci -p <ime> \"tekst\"{X}\n")
        print("  Tiha soba, mikrofon na razdaljinu raširene šake, malo sa strane")
        print("  da plozivi ne udaraju pravo u membranu.\n")
        ask("ENTER za izbor mikrofona… ")
        print()
        src = pick_source()
        calibrate(src, tmp)
        ref = record_passage(src, tmp, passage)

    while True:
        test = ask(f"\n{B}Šta da izgovori za probu?{X} [ENTER = isti tekst] ") \
               or " ".join(passage.split())[:120]
        print(f"{D}· šaljem i generišem…{X}")
        try:
            res = cli.speak_with_ref(test, ref, " ".join(passage.split())) \
                if hasattr(cli, "speak_with_ref") else None
        except Exception:
            res = None
        if res is None:
            # No throwaway-reference endpoint: enrol under a temporary name,
            # audition it, and rename or drop it depending on the verdict.
            tmp_name = "__cvoice_probe__"
            try:
                cli.enrol(tmp_name, ref, " ".join(passage.split()), notes="probe")
                res = cli.speak(test, profile=tmp_name, takes=2)
            except ServerError as exc:
                print(f"{R}{exc}{X}", file=sys.stderr); return 1
        out = tmp / "test.wav"
        out.write_bytes(res["audio"])
        for s in (res.get("scores") or [])[:2]:
            print(f"  {D}WER {s['wer']:.1f}%{X}  {s['transcript'][:64]}")
        audio.play(out)

        k = ask(f"\n{B}Valja?{X} [d = da, p = pusti opet, t = drugi tekst, "
                f"r = snimi ponovo, q = odustani] ").lower()
        if k in ("d", "da", "y", ""):
            break
        if k == "p":
            audio.play(out); continue
        if k == "t":
            continue
        if k == "r":
            if src is None:
                print(f"{Y}--from-wav režim: ne mogu da snimim ponovo{X}"); continue
            ref = record_passage(src, tmp, passage); continue
        if k == "q":
            try:
                cli.delete("__cvoice_probe__")
            except Exception:
                pass
            print("odustao."); return 0

    name = args.name or ask(f"\n{B}Ime profila?{X} (npr. Sanja Petrović) ")
    if not name:
        print(f"{R}bez imena ne mogu da snimim{X}"); return 1

    existing = {p["slug"] for p in cli.profiles()}
    from .profiles import slugify
    slug = slugify(name)
    if slug in existing:
        if ask(f"{Y}Profil '{slug}' već postoji. Prepisati? [d/N] {X}").lower() not in ("d", "y"):
            print("prekinuto."); return 0

    try:
        res = cli.enrol(name, ref, " ".join(passage.split()), notes="cvoice")
        cli.delete("__cvoice_probe__")
    except ServerError as exc:
        print(f"{R}{exc}{X}", file=sys.stderr); return 1
    except Exception:
        pass

    meta = res.get("profile", {})
    print(f"\n{G}{B}Sačuvano.{X}  {B}{meta.get('name', name)}{X}  "
          f"{D}({meta.get('slug', slug)}, {meta.get('duration', 0)}s){X}\n")
    print(f"  {B}reci -p {meta.get('slug', slug)} \"tekst\"{X}")
    if src is not None:
        g = audio.gain_db()
        if g is not None:
            print(f"{D}  mikrofon ostavljen na {g:.1f} dB{X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
