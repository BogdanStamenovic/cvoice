# -*- coding: utf-8 -*-
"""Turn a URL or an existing recording into a reference clip.

The output contract is the same one the microphone path produces: 48 kHz mono
WAV, peak near -9 dBFS, no clipping. Anything that reaches the enrolment step
has been through here, so a downloaded clip and a recorded one are treated
identically from that point on.

Denoising is offered but deliberately gentle. The speaker encoder keys on
high-frequency detail, and aggressive noise reduction strips some of that out
along with the noise - a lightly noisy but full-band clip often clones better
than a heavily processed one. When --denoise is used the untouched version is
kept alongside so the two can be compared.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .audio import stats

TARGET_PEAK_DBFS = -9.0


def parse_time(value) -> float | None:
    """Accept 18, '18', '1:24', '00:01:24.5'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip()
    if not v:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return float(v)
    parts = v.split(":")
    if not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"nerazumljivo vreme: {value}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def _fmt(seconds: float) -> str:
    return "%02d:%02d:%06.3f" % (int(seconds // 3600), int(seconds % 3600 // 60), seconds % 60)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def ytdlp_cmd():
    """Prefer a yt-dlp on PATH; fall back to the module in this venv."""
    if have("yt-dlp"):
        return ["yt-dlp"]
    import sys
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        return None


def fetch_url(url: str, workdir: Path, start=None, duration=None, log=print) -> Path:
    """Download audio, fetching only the wanted section where the site allows it."""
    cmd = ytdlp_cmd()
    if cmd is None:
        raise RuntimeError(
            "yt-dlp nije dostupan — instaliraj ga sa: pip install yt-dlp")
    out = workdir / "source.%(ext)s"
    argv = [*cmd, "--no-playlist", "-f", "bestaudio/best",
            "-x", "--audio-format", "wav", "--audio-quality", "0",
            "-o", str(out), "--no-progress", "--quiet"]
    s = parse_time(start)
    d = parse_time(duration)
    if s is not None and d is not None:
        # Ask the server for just this window instead of pulling the whole
        # video and throwing 99% of it away.
        argv += ["--download-sections", f"*{_fmt(s)}-{_fmt(s + d)}",
                 "--force-keyframes-at-cuts"]
    argv.append(url)
    log("· preuzimam zvuk…")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    hits = sorted(workdir.glob("source.*"))
    if proc.returncode != 0 or not hits:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError("preuzimanje nije uspelo: " +
                           (msg[-1] if msg else "nepoznata greška"))
    return hits[0]


def prepare(src: Path, dst: Path, start=None, duration=None, denoise=False,
            trimmed_by_download=False, log=print) -> dict:
    """Trim, downmix, optionally denoise, and normalise the peak.

    Returns the stats of the finished file.
    """
    if not have("ffmpeg"):
        raise RuntimeError("ffmpeg nije dostupan")

    s = parse_time(start)
    d = parse_time(duration)
    pre = []
    if not trimmed_by_download:
        if s is not None:
            pre += ["-ss", _fmt(s)]
        if d is not None:
            pre += ["-t", _fmt(d)]

    chain = []
    if denoise:
        # These numbers are measured, not chosen by feel. On a reference clip
        # with pink noise mixed in, sweeping afftdn gave:
        #
        #   setting                    SNR    >8kHz    bandwidth
        #   (noisy input)             12.2    3.65%     16.3 kHz
        #   nr=12:nf=-30              22.1    1.77%     12.3 kHz
        #   nr=8:nf=-35               19.8    2.75%     13.4 kHz
        #   nr=6:nf=-40:tn=1          18.1    3.74%     14.5 kHz
        #
        # nr=12 buys the most SNR by cutting 4 kHz off the top and halving the
        # high-frequency content - which is exactly what the speaker encoder
        # keys on, so it trades away the thing being cloned. nr=6 with noise
        # tracking keeps almost all the SNR gain and leaves the highs intact.
        # Do not raise nr without re-measuring the >8 kHz column.
        chain += ["highpass=f=60", "afftdn=nr=6:nf=-40:tn=1"]
    filters = ",".join(chain) if chain else None

    tmp = dst.with_suffix(".stage.wav")
    argv = ["ffmpeg", "-v", "error", "-y", *pre, "-i", str(src),
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le"]
    if filters:
        argv += ["-af", filters]
    argv.append(str(tmp))
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError("ffmpeg: " + (proc.stderr or "").strip()[-300:])

    # Second pass for level: measure the real peak, then apply one fixed gain.
    # Compressing or auto-levelling here would flatten the dynamics the encoder
    # is meant to learn.
    st = stats(tmp)
    if st["duration"] < 0.2:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("isečak je prazan — proveri --start/--duration")
    gain = TARGET_PEAK_DBFS - st["peak_dbfs"]
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(tmp),
         "-af", "volume=%.2fdB" % gain, "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True, timeout=600)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError("ffmpeg (nivo): " + (proc.stderr or "").strip()[-300:])
    final = stats(dst)
    log("· %s%.1f s · vrh %.1f dBFS · klipovano %.3f%%"
        % ("denoise · " if denoise else "", final["duration"],
           final["peak_dbfs"], final["clipped_pct"]))
    return final


def ingest(source: str, workdir: Path, start=None, duration=None,
           denoise=False, log=print):
    """URL or local file -> (reference wav, optional un-denoised twin)."""
    workdir.mkdir(parents=True, exist_ok=True)
    is_url = bool(re.match(r"^[a-z][a-z0-9+.-]*://", str(source), re.I))
    trimmed = False
    if is_url:
        raw = fetch_url(source, workdir, start, duration, log=log)
        trimmed = parse_time(start) is not None and parse_time(duration) is not None
    else:
        raw = Path(source).expanduser()
        if not raw.exists():
            raise RuntimeError(f"nema fajla: {raw}")

    ref = workdir / "ref.wav"
    prepare(raw, ref, start, duration, denoise=denoise,
            trimmed_by_download=trimmed, log=log)

    twin = None
    if denoise:
        # Keep the unprocessed version so the two can be auditioned against
        # each other - denoising is a trade, not a free improvement.
        twin = workdir / "ref-raw.wav"
        try:
            prepare(raw, twin, start, duration, denoise=False,
                    trimmed_by_download=trimmed, log=lambda *_: None)
        except RuntimeError:
            twin = None
    return ref, twin
