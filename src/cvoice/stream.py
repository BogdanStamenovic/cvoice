"""Streaming speech: text in pieces, audio out in pieces.

**Why chunks and not frames.** OmniVoice is non-autoregressive: it predicts the
whole clip's length, then unmasks every token at once over N steps. There is no
point at which the first 200 ms exists before the rest does, so streaming here
means rendering the text a clause at a time and sending each clause as soon as
it is done. The library does the same thing internally for long text
(`_generate_chunked`, each chunk conditioned on the same reference), so
independently rendered clauses are the model's own idea of how to go long.

**Why the first chunk is short and the rest are not.** Time to first audio is
set by the first chunk alone, and a short clause renders in ~0.4 s at 8 steps.
After that, rendering runs ~10x faster than playback (a 5 s sentence in 0.49 s,
measured 2026-10-30 on the 4060), so later chunks can afford to be whole
sentences, which means fewer seams and better prosody.

**Why 8 steps by default.** Measured 2026-10-30, 45 clips, Whisper large-v3 CER
(mean / worst): 32 steps 8.2/20, 16 steps 4.9/10, 8 steps 6.6/25, 6 steps
11.4/50, 4 steps 17.2/43. 32, 16 and 8 are within Whisper's own noise; 6 and
below garble words. Wall time: 1.52 s / 0.77 s / 0.39 s for the same short line.
"""

from __future__ import annotations

import base64

import numpy as np

DEFAULT_STEPS = 8

FIRST_MIN_WORDS = 2
"""The first chunk may end at any clause mark once it has this many words."""
FIRST_MAX_CHARS = 60
"""...and is cut at a space once it gets this long with no clause mark at all."""
MIN_CHARS = 40
"""Later chunks end at a sentence mark only once they are this long."""
COMMA_CHARS = 120
"""Later chunks fall back to a comma once they are this long."""
MAX_CHARS = 240
"""Hard cap: cut at the last space. Long chunks cost VRAM, not just time."""

GEN = {"pad_duration": 0.0, "fade_duration": 0.01}
"""Render settings for a chunk. OmniVoice's defaults pad 100 ms of silence and a
100 ms fade onto BOTH ends of every clip, which is right for a standalone file
and wrong between clauses: on the first live call (2026-10-30) the seams
measured 450-750 ms of silence and he heard every one ("prekid između
segmenata... pa onda build up"). 10 ms of fade is kept so a cut edge does not
click."""

EDGE_KEEP = 0.02
"""Seconds of quiet kept at each edge after trimming, so a soft onset survives."""

PAUSE_AFTER = {"clause": 0.10, "sentence": 0.25}
"""The pause put back after a chunk, by how it ended. Roughly what a speaker
leaves at a comma and at a full stop; the last chunk of a reply gets none."""

SENTENCE = ".!?…"
CLAUSE = SENTENCE + ",;:—–"


class Segmenter:
    """Turns a stream of text deltas into speakable chunks.

    A mark only counts as a boundary once the character AFTER it has arrived
    and is whitespace. That is what stops "3.5" or "v1.2" being cut in half,
    at the price of holding the last chunk until the next delta or `flush()`.
    """

    def __init__(self):
        self.buf = ""
        self.emitted = 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (cut := self._cut()) is not None:
            piece, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece, self.buf = " ".join(self.buf.split()), ""
        if piece:
            self.emitted += 1
            return [piece]
        return []

    def _boundaries(self, marks: str):
        for i, ch in enumerate(self.buf[:-1]):
            if ch in marks and self.buf[i + 1].isspace():
                yield i + 1

    def _cut(self) -> int | None:
        buf = self.buf
        if self.emitted == 0:
            for end in self._boundaries(CLAUSE):
                if len(buf[:end].split()) >= FIRST_MIN_WORDS:
                    return end
            if len(buf) >= FIRST_MAX_CHARS:
                return _last_space(buf, FIRST_MAX_CHARS)
            return None
        for end in self._boundaries(SENTENCE):
            if end >= MIN_CHARS:
                return end
        if len(buf) >= COMMA_CHARS:
            commas = [e for e in self._boundaries(CLAUSE) if MIN_CHARS <= e <= MAX_CHARS]
            if commas:
                return commas[-1]
        if len(buf) >= MAX_CHARS:
            return _last_space(buf, MAX_CHARS)
        return None


def _last_space(buf: str, limit: int) -> int:
    cut = buf.rfind(" ", 0, limit)
    return cut if cut > 0 else limit


def split(text: str) -> list[str]:
    """The whole-text case: the same chunks a token stream of `text` would give."""
    seg = Segmenter()
    return seg.feed(text + " ") + seg.flush()


def tidy(audio, samplerate: int, text: str, last: bool) -> np.ndarray:
    """Trim a rendered chunk to its speech, then add the pause its ending earns.

    The level gate is relative (2% of the chunk's own peak), so a quiet profile
    is trimmed the same as a loud one."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a
    frame = max(1, int(samplerate * 0.01))
    n = a.size // frame
    if n:
        level = np.abs(a[: n * frame]).reshape(n, frame).max(axis=1)
        loud = np.flatnonzero(level > 0.02 * float(level.max() or 1.0))
        if loud.size:
            keep = int(EDGE_KEEP * samplerate)
            start = max(0, loud[0] * frame - keep)
            end = min(a.size, (loud[-1] + 1) * frame + keep)
            a = a[start:end]
    if last:
        return a
    ending = text.rstrip()[-1:] if text.strip() else ""
    pause = PAUSE_AFTER["sentence" if ending in SENTENCE else "clause"]
    return np.concatenate([a, np.zeros(int(pause * samplerate), dtype=np.float32)])


def pcm16(audio) -> bytes:
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def chunk_message(seq: int, text: str, audio, samplerate: int, gen_s: float) -> dict:
    """One rendered chunk on the wire: mono little-endian int16 PCM, base64.

    JSON rather than raw frames so a curl or a websocket console can read it;
    the base64 overhead is ~33% of 48 kB/s, which no link this runs over notices.
    """
    raw = pcm16(audio)
    return {"seq": seq, "text": text, "samplerate": samplerate,
            "samples": len(raw) // 2, "duration": round(len(raw) / 2 / samplerate, 3),
            "gen_s": round(gen_s, 3), "pcm_b64": base64.b64encode(raw).decode()}


def decode_chunk(msg: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(msg["pcm_b64"]), dtype="<i2").astype(np.float32) / 32767.0

