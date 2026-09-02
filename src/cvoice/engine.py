# -*- coding: utf-8 -*-
"""OmniVoice wrapper. Loads on first use and stays warm.

Lazy loading is the point of running a daemon at all: the model costs a few
seconds to load and ~2.2 GiB of VRAM to hold, so paying that once per process
rather than once per sentence is the whole reason this is a server.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def _is_cached(cache_dir) -> bool:
    """A present directory is not a cached model - huggingface_hub leaves a
    metadata stub behind after a bare API call, and an interrupted download
    leaves a partial one. Judge by weight, not existence."""
    if cache_dir is None or not cache_dir.exists():
        return False
    return _dir_bytes(cache_dir) > 50_000_000


def _hf_cache_dir(model_id: str):
    """Where huggingface_hub will put this repo, or None for a local path."""
    if os.sep in str(model_id) and Path(model_id).exists():
        return None
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return home / "hub" / ("models--" + str(model_id).strip("/").replace("/", "--"))


def _dir_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _repo_bytes(model_id: str):
    """Total download size, so progress can be a fraction rather than a number
    that just goes up. Best-effort - the API call is not worth failing over."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, files_metadata=True)
        return sum(f.size or 0 for f in (info.siblings or [])) or None
    except Exception:
        return None


class Engine:
    SAMPLE_RATE = 24000

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        # Read by GET /status so the client can show what a long first call is
        # actually doing - a silent 3 GB download is indistinguishable from a hang.
        self.status = {"phase": "idle", "detail": "", "downloaded": 0,
                       "total": None, "rate": 0.0}
        self._watch_stop = None

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

    def _watch_download(self, cache_dir: Path, total):
        """Sample the cache directory instead of hooking huggingface_hub's
        internals: it works the same whether the transfer backend is hf_transfer,
        plain requests, or a resumed partial download."""
        start = time.time()
        base = _dir_bytes(cache_dir) if cache_dir.exists() else 0
        while not self._watch_stop.is_set():
            now = _dir_bytes(cache_dir) if cache_dir.exists() else 0
            elapsed = max(time.time() - start, 0.001)
            self.status.update(phase="downloading", downloaded=now, total=total,
                               rate=(now - base) / elapsed,
                               detail=os.path.basename(str(cache_dir)))
            self._watch_stop.wait(1.0)

    def load(self):
        """Idempotent. First call downloads the weights if they are not cached."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            import torch
            from omnivoice import OmniVoice

            cache_dir = _hf_cache_dir(self.model_id)
            watcher = None
            if cache_dir is not None and not _is_cached(cache_dir):
                self.status.update(phase="downloading", detail=str(self.model_id),
                                   downloaded=0, total=None, rate=0.0)
                self._watch_stop = threading.Event()
                watcher = threading.Thread(
                    target=self._watch_download,
                    args=(cache_dir, _repo_bytes(self.model_id)), daemon=True)
                watcher.start()
            else:
                self.status.update(phase="loading", detail=str(self.model_id))

            dev = self._resolve_device()
            # fp16 is a clear win on CUDA; on MPS it is still patchy, so stay in
            # fp32 there and on CPU.
            dtype = torch.float16 if dev.startswith("cuda") else torch.float32
            try:
                try:
                    self._model = OmniVoice.from_pretrained(
                        self.model_id, device_map=dev, dtype=dtype)
                except Exception:
                    if dev == "cpu":
                        raise
                    self._model = OmniVoice.from_pretrained(
                        self.model_id, device_map="cpu", dtype=torch.float32)
                    dev = "cpu"
            finally:
                if watcher is not None:
                    self._watch_stop.set()
                    watcher.join(timeout=2)
            self.device_used = dev
            self.status.update(phase="ready", detail=dev, rate=0.0)
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
