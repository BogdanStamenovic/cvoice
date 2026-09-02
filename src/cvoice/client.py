# -*- coding: utf-8 -*-
"""Thin HTTP client. Deliberately free of torch so it installs on any laptop."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx


class ServerError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str, token: str = "", timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _check(self, r):
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ServerError(f"{r.status_code}: {detail}")
        return r

    def health(self) -> dict:
        with httpx.Client(timeout=15.0) as c:
            return self._check(c.get(f"{self.base}/health")).json()

    def passage(self, lang: str | None = None) -> str:
        with httpx.Client(timeout=15.0) as c:
            url = f"{self.base}/passage" + (f"?lang={lang}" if lang else "")
            return self._check(c.get(url)).json().get("text", "")

    def status(self) -> dict:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{self.base}/status")
            return r.json() if r.status_code < 400 else {}

    def profiles(self) -> list[dict]:
        with httpx.Client(timeout=30.0) as c:
            r = self._check(c.get(f"{self.base}/profiles", headers=self._headers()))
            return r.json().get("profiles", [])

    def enrol(self, name: str, wav_path, text: str, notes: str = "") -> dict:
        with httpx.Client(timeout=self.timeout) as c:
            with open(wav_path, "rb") as fh:
                r = c.post(f"{self.base}/profiles", headers=self._headers(),
                           data={"name": name, "text": text, "notes": notes},
                           files={"file": (Path(wav_path).name, fh, "audio/wav")})
            return self._check(r).json()

    def delete(self, slug: str) -> dict:
        with httpx.Client(timeout=30.0) as c:
            return self._check(c.delete(f"{self.base}/profiles/{slug}",
                                        headers=self._headers())).json()

    def speak(self, text: str, profile: str = "", takes: int = 3,
              language: str | None = None) -> dict:
        body = {"text": text, "takes": takes}
        if profile:
            body["profile"] = profile
        if language:
            body["language"] = language
        with httpx.Client(timeout=self.timeout) as c:
            r = self._check(c.post(f"{self.base}/speak", json=body, headers=self._headers()))
        out = r.json()
        out["audio"] = base64.b64decode(out.pop("audio_b64"))
        return out
