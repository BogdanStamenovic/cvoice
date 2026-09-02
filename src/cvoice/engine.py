# -*- coding: utf-8 -*-
"""OmniVoice wrapper. Loads on first use and stays warm.

Lazy loading is the point of running a daemon at all: the model costs a few
seconds to load and ~2.2 GiB of VRAM to hold, so paying that once per process
rather than once per sentence is the whole reason this is a server.
"""
from __future__ import annotations

import threading
from pathlib import Path


class Engine:
    SAMPLE_RATE = 24000

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            # Apple Silicon. Not as fast as CUDA and some ops still fall back to
            # the CPU, but far better than pure CPU - and load() drops back to
            # CPU if any of it turns out to be unsupported.
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Idempotent. First call downloads the weights if they are not cached."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            import torch
            from omnivoice import OmniVoice

            dev = self._resolve_device()
            # fp16 is a clear win on CUDA; on MPS it is still patchy, so stay in
            # fp32 there and on CPU.
            dtype = torch.float16 if dev.startswith("cuda") else torch.float32
            try:
                self._model = OmniVoice.from_pretrained(
                    self.model_id, device_map=dev, dtype=dtype)
            except Exception:
                if dev == "cpu":
                    raise
                self._model = OmniVoice.from_pretrained(
                    self.model_id, device_map="cpu", dtype=torch.float32)
                dev = "cpu"
            self.device_used = dev
            return self._model

    def speak(self, text: str, ref_wav: str, ref_text: str, seed: int | None = None):
        import torch

        model = self.load()
        # Generation is not thread-safe and one GPU cannot overlap requests
        # usefully anyway, so serialise here rather than in every caller.
        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            return model.generate(text=text, ref_audio=ref_wav, ref_text=ref_text)[0]
