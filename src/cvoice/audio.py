# -*- coding: utf-8 -*-
"""Microphone capture, level calibration and playback.

Capture goes through PortAudio (sounddevice) so the same code works on Linux,
macOS and Windows. Input *gain* is a different matter: only Linux exposes a
portable way to set it (pactl), so elsewhere the level is measured and the user
is told what to change. Either way a clipped take is refused.

Why any of this exists: on the machine this was built for, PipeWire had the mic
at 100% while its hardware 0 dB point was 20% (-42 dB) - about 42 dB of software
gain on the capsule. That clipped 4-12% of samples with flat-topped runs up to
82 samples long. A speaker encoder learns that distortion as part of the timbre,
and it cannot be repaired afterwards: ffmpeg's declipper recovered 2 samples out
of 63,057.
"""
from __future__ import annotations

import math
import platform
import re
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 48000
TARGET_PEAK_DBFS = -9.0
ACCEPT_LO, ACCEPT_HI = -16.0, -3.0
CLIP_TOLERANCE_PCT = 0.02

LINUX = platform.system() == "Linux"


# --------------------------------------------------------------- device list
def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception:
        return None


def have_capture() -> bool:
    return _sd() is not None


def sources() -> list[dict]:
    """[{index, name, channels, default}] for every input device."""
    sd = _sd()
    if sd is None:
        return []
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) < 1:
            continue
        out.append({"index": i, "name": d["name"],
                    "channels": d["max_input_channels"],
                    "default": (i == default_in)})
    return out


def is_bluetooth(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ("bluez", "bluetooth", "hands-free", "hfp", "airpods"))


def describe(dev: dict) -> str:
    n = dev["name"].lower()
    if is_bluetooth(n):
        return "bluetooth — odbijeno, HFP/mSBC seče visoke frekvencije"
    if "usb" in n:
        return "USB"
    if any(k in n for k in ("built-in", "internal", "analog", "macbook")):
        return "ugrađeni"
    return ""


# ------------------------------------------------------------------ capture
def record(path, device_index: int | None, seconds: float) -> None:
    import numpy as np
    import soundfile as sf
    sd = _sd()
    if sd is None:
        raise RuntimeError("sounddevice nije instaliran")
    frames = int(seconds * SAMPLE_RATE)
    buf = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                 dtype="int16", device=device_index)
    sd.wait()
    sf.write(str(path), np.asarray(buf).reshape(-1), SAMPLE_RATE, subtype="PCM_16")


def record_until_enter(path, device_index: int | None, prompt: str = "") -> None:
    """Record until the user presses ENTER. Unbounded, so it streams to a file
    rather than filling memory."""
    import soundfile as sf
    sd = _sd()
    if sd is None:
        raise RuntimeError("sounddevice nije instaliran")
    with sf.SoundFile(str(path), mode="w", samplerate=SAMPLE_RATE,
                      channels=1, subtype="PCM_16") as fh:
        def cb(indata, frames, time_info, status):
            fh.write(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=device_index, callback=cb):
            try:
                input(prompt)
            except (EOFError, KeyboardInterrupt):
                pass


def play(path) -> None:
    import soundfile as sf
    sd = _sd()
    if sd is not None:
        try:
            data, sr = sf.read(str(path), dtype="float32")
            sd.play(data, sr)
            sd.wait()
            return
        except Exception:
            pass
    for cmd in (["pw-play", str(path)], ["paplay", str(path)],
                ["afplay", str(path)],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, capture_output=True, timeout=600)
            return


# ------------------------------------------------------- measurement / gain
def stats(path) -> dict:
    """Peak, RMS, duration and clipped-sample percentage, read from the PCM.

    Deliberately not ffmpeg's volumedetect: ffmpeg 9.x prints those stats at
    INFO level, so the obvious `-v error` invocation returns an empty string and
    the meter silently reads blank.
    """
    with wave.open(str(path), "rb") as w:
        n, sr, sw = w.getnframes(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2 or n == 0:
        return {"duration": 0.0, "peak_dbfs": -99.0, "rms_dbfs": -99.0, "clipped_pct": 0.0}
    a = array("h")
    a.frombytes(raw)
    peak = max((abs(v) for v in a), default=0)
    clipped = sum(1 for v in a if abs(v) >= 32766)
    rms = math.sqrt(sum(float(v) * v for v in a) / len(a)) if a else 0.0
    to_db = lambda x: 20 * math.log10(x / 32768.0) if x > 0 else -99.0
    return {"duration": n / float(sr), "peak_dbfs": to_db(peak),
            "rms_dbfs": to_db(rms), "clipped_pct": 100.0 * clipped / len(a)}


def gain_controllable() -> bool:
    return LINUX and shutil.which("pactl") is not None


def _pactl_default_source() -> str | None:
    try:
        r = subprocess.run(["pactl", "get-default-source"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def gain_db(source: str | None = None) -> float | None:
    if not gain_controllable():
        return None
    source = source or _pactl_default_source()
    if not source:
        return None
    try:
        out = subprocess.run(["pactl", "list", "sources"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return None
    block = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Name:"):
            block = s.split("Name:", 1)[1].strip() == source
        elif block and s.startswith("Volume:"):
            m = re.search(r"(-?[\d.]+) dB", line)
            if m:
                return float(m.group(1))
    return None


def set_gain_db(db: float, source: str | None = None) -> bool:
    """pactl treats a leading '-' as a RELATIVE change, so passing '-18dB' would
    subtract 18 from the current value and a feedback loop diverges rather than
    converging. Volume is cubic - dB = 60*log10(raw/65536) - so set a raw int."""
    if not gain_controllable():
        return False
    source = source or _pactl_default_source()
    if not source:
        return False
    raw = max(1, min(int(round(65536 * (10 ** (db / 60.0)))), 65536))
    try:
        subprocess.run(["pactl", "set-source-volume", source, str(raw)],
                       capture_output=True, timeout=15)
        return True
    except Exception:
        return False


def silence_hint() -> str:
    """Why a recording came back empty. On macOS a terminal without microphone
    permission records digital silence instead of raising, so the honest first
    guess there is permissions, not a muted device."""
    if platform.system() == "Darwin":
        return ("macOS traži dozvolu za mikrofon: System Settings → Privacy & "
                "Security → Microphone → uključi za svoj terminal, pa ga restartuj")
    if platform.system() == "Windows":
        return ("Settings → Privacy & security → Microphone → dozvoli pristup "
                "desktop aplikacijama")
    return "proveri da mikrofon nije utišan i da je izabran pravi uređaj"


def manual_gain_hint() -> str:
    if platform.system() == "Darwin":
        return "System Settings → Sound → Input → smanji Input volume"
    if platform.system() == "Windows":
        return "Settings → System → Sound → tvoj mikrofon → smanji Input volume"
    return "smanji ulazno pojačanje u zvučnim podešavanjima"
