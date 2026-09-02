# -*- coding: utf-8 -*-
"""Optional take-scoring with Whisper.

Short lines are where the model is flakiest - on a four-word sentence two of
three takes came back wrong in testing (wrong vocative, slurred words). So the
default is to generate several and keep the one that transcribes closest.

Installed via the `asr` extra; if it is missing, scoring silently degrades to
"return the first take", which is still a working tool.
"""
from __future__ import annotations

import pathlib
import re
import threading
import unicodedata

_CYR = "абвгдђежзијклљмнњопрстћуфхцчџш"
_LAT = ["a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "lj",
        "m", "n", "nj", "o", "p", "r", "s", "t", "ć", "u", "f", "h", "c", "č",
        "dž", "š"]


def normalise(s: str) -> str:
    """Fold the differences that do not matter for judging a take."""
    s = "".join(_LAT[_CYR.index(c)] if c in _CYR else c for c in (s or "").lower())
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("dj", "d"), ("dz", "z"), ("nj", "n"), ("lj", "l")):
        s = s.replace(a, b)
    # Deliberate vowel stretching ("Sanjaaa") is a phrasing device, not an error.
    s = re.sub(r"([aeiou])\1+", r"\1", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def _preload_cuda_libs() -> None:
    """Make ctranslate2 find the CUDA libraries that torch vendored.

    faster-whisper/ctranslate2 dlopen libcublas and libcudnn by soname and only
    search the linker path. In a venv those libraries live under
    site-packages/nvidia/*/lib, which is not on it - and LD_LIBRARY_PATH cannot
    be fixed from inside a running process. Loading them explicitly with
    RTLD_GLOBAL puts them in the process before ctranslate2 asks, which works
    wherever the env-var trick would have.
    """
    import ctypes
    import glob
    import os
    import sys

    if os.name != "posix":
        return
    roots = [pathlib.Path(p) / "nvidia" for p in sys.path if p.endswith("site-packages")]
    # cublas needs cublasLt present first; cudnn pulls its own ordered set.
    order = ("libcudart", "libcublasLt", "libcublas", "libcudnn")
    for stem in order:
        for root in roots:
            for hit in sorted(glob.glob(str(root / "*/lib" / (stem + ".so*")))):
                try:
                    ctypes.CDLL(hit, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    continue


class Scorer:
    def __init__(self, model_name: str = "large-v3", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            import jiwer  # noqa: F401
            return True
        except Exception:
            return False

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            dev = self._resolve_device()
            if dev == "cuda":
                _preload_cuda_libs()
            from faster_whisper import WhisperModel
            try:
                self._model = WhisperModel(
                    self.model_name, device=dev,
                    compute_type="float16" if dev == "cuda" else "int8")
            except Exception:
                # A CUDA runtime mismatch should cost speed, not the feature.
                if dev != "cpu":
                    self._model = WhisperModel(self.model_name, device="cpu",
                                               compute_type="int8")
                else:
                    raise
            return self._model

    def score(self, wav_paths, text: str, language: str = "sr"):
        """Return [(wer, cer, path, transcript)] sorted best-first."""
        from jiwer import cer, wer

        model = self.load()
        ref = normalise(text)
        rows = []
        with self._lock:
            for p in wav_paths:
                segs, _ = model.transcribe(str(p), language=language, beam_size=5)
                hyp = " ".join(s.text for s in segs).strip()
                h = normalise(hyp)
                rows.append((wer(ref, h) * 100 if ref else 0.0,
                             cer(ref, h) * 100 if ref else 0.0, str(p), hyp))
        return sorted(rows, key=lambda r: (r[0], r[1]))
