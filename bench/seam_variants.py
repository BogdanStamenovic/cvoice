"""Render one reply through each chunk-joining approach, for listening side by side.

Every variant uses the same text, seed and step count; only how chunks are cut
and joined differs. "whole" is the unchunked render, as the reference for what
the voice sounds like with no seams at all.

    .venv/bin/python bench/seam_variants.py OUT_DIR
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvoice import stream  # noqa: E402
from cvoice.engine import Engine  # noqa: E402
from cvoice.profiles import Store  # noqa: E402

SR = 24000
TEXT = ("Samo sekund, da proverim. Poslednja izmena u cvoice repou je pre tri minuta, "
        "i odnosi se na pauze između delova. Sve ostalo radi kako treba, a kiri je "
        "pauziran dok traje poziv. Javi mi kako ti zvuči.")
SEED, STEPS = 4000, 8


def _ending_pause(text):
    return stream.PAUSE_AFTER["sentence" if text.rstrip()[-1:] in stream.SENTENCE else "clause"]


def v2_trim(a, text, last):
    """Call 2 (2026-10-30): trim to 2% of peak, keep 20 ms, pause of zeros."""
    f = SR // 100
    n = a.size // f
    lvl = np.abs(a[: n * f]).reshape(n, f).max(1)
    loud = np.flatnonzero(lvl > 0.02 * lvl.max())
    keep = int(0.02 * SR)
    a = a[max(0, loud[0] * f - keep): min(a.size, (loud[-1] + 1) * f + keep)]
    return a if last else np.concatenate([a, np.zeros(int(_ending_pause(text) * SR), np.float32)])


def v3_loop(a, text, last):
    """Call 3 (2026-10-31): strip zeros, pause = quietest 80 ms looped. The buzzsaw."""
    live = np.flatnonzero(np.abs(a) > 1e-4)
    a = a[live[0]: live[-1] + 1]
    if last:
        return a
    n = int(_ending_pause(text) * SR)
    w = int(0.08 * SR)
    best = min(range(0, a.size - w + 1, w // 4), key=lambda s: np.mean(a[s:s + w] ** 2))
    seg = a[best:best + w] * min(3.0, stream._floor(a, SR) / max(1e-9, np.sqrt(np.mean(a[best:best + w] ** 2))))
    x = w // 4
    ramp = np.linspace(0, 1, x, dtype=np.float32)
    out = seg.copy()
    while out.size < n:
        out = np.concatenate([out[:-x], out[-x:] * ramp[::-1] + seg[:x] * ramp, seg[x:]])
    return np.concatenate([a, out[:n]])


def v4_synth(a, text, last):
    """Now: strip zeros, pause = synthesised noise shaped like the chunk's background."""
    return stream.tidy(a, SR, text, last)


def v5_tailcut(a, text, last):
    """v4, plus the decaying tail after the last clear speech cut (keep 80 ms),
    crossfaded over 30 ms into the synthesised background."""
    live = np.flatnonzero(np.abs(a) > 1e-4)
    a = a[live[0]: live[-1] + 1]
    f = SR // 100
    n = a.size // f
    fl = stream._floor(a, SR)
    rms = np.sqrt((a[: n * f].reshape(n, f) ** 2).mean(1))
    speech = np.flatnonzero(rms > 2.5 * fl)
    if speech.size:
        a = a[: min(a.size, (speech[-1] + 1) * f + int(0.08 * SR))]
    pause_s = 0.0 if last else _ending_pause(text)
    x = int(0.03 * SR)
    tone = stream.room_tone(a, SR, int(pause_s * SR) + x)
    ramp = np.linspace(0, 1, x, dtype=np.float32)
    joined = a[-x:] * ramp[::-1] + tone[:x] * ramp
    out = np.concatenate([a[:-x], joined, tone[x:]])
    if last:
        out[-x:] *= ramp[::-1]
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.load(open(Path.home() / ".config/cvoice/config.toml", "rb"))
    engine = Engine(cfg["server"]["model"])
    store = Store(Path(cfg["server"]["data_dir"]))
    prof = store.load(store.resolve(cfg["client"]["profile"]))

    def render(text, **gen):
        return np.asarray(engine.speak(text, prof["wav"], prof["text"], seed=SEED,
                                       steps=STEPS, **gen), dtype=np.float32).reshape(-1)

    sf.write(out / "0-whole-no-chunks.wav", render(TEXT), SR)
    clause_chunks = stream.split(TEXT)
    sentence_chunks = [s.strip() + "." for s in TEXT.rstrip(".").split(". ")]
    raw_default = [render(c) for c in clause_chunks]
    raw_bare = [render(c, **stream.GEN) for c in clause_chunks]
    raw_bare_sent = [render(c, **stream.GEN) for c in sentence_chunks]

    sf.write(out / "1-call1-library-padding.wav", np.concatenate(raw_default), SR)
    for name, fn, raws, chunks in (
        ("2-call2-trim-zero-pauses", v2_trim, raw_bare, clause_chunks),
        ("3-call3-looped-tone-buzzsaw", v3_loop, raw_bare, clause_chunks),
        ("4-synth-tone", v4_synth, raw_bare, clause_chunks),
        ("5-synth-tone-tailcut", v5_tailcut, raw_bare, clause_chunks),
        ("6-synth-tone-tailcut-sentences-only", v5_tailcut, raw_bare_sent, sentence_chunks),
    ):
        parts = [fn(a, t, i == len(chunks) - 1) for i, (a, t) in enumerate(zip(raws, chunks))]
        sf.write(out / f"{name}.wav", np.concatenate(parts), SR)
    print("clause chunks:", clause_chunks)
    print("sentence chunks:", sentence_chunks)


if __name__ == "__main__":
    main()
