# -*- coding: utf-8 -*-
"""Thin HTTP client. Deliberately free of torch so it installs on any laptop."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx


class ServerError(RuntimeError):
    pass


class Client:
    """Talks to cvoiced.

    `fallback` exists because the server binds ONE address and the client is
    configured with another. The daemon listens on the tailnet address so the
    thin client can reach it from a laptop; a client running on the same box as
    the server therefore points at loopback and gets ECONNREFUSED from a daemon
    that is running perfectly well. Rather than hardcode the tailnet address --
    which then breaks whenever Tailscale is down -- the first connection error
    retries once against the server's own configured bind address and sticks
    with whichever answered.
    """

    def __init__(self, base_url: str, token: str = "", timeout: float = 600.0,
                 fallback: str = ""):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.fallback = fallback.rstrip("/")

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

    def _try(self, call):
        """Run `call(base)`, retrying once against the fallback if nothing answers.

        Only connection-level failures fall through. A 4xx/5xx means we reached
        the right daemon and it said no, which retrying elsewhere would hide.
        """
        try:
            return call(self.base)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            if not self.fallback or self.fallback == self.base:
                raise
            result = call(self.fallback)
            self.base = self.fallback  # only after it actually worked
            return result

    def health(self) -> dict:
        def go(base):
            with httpx.Client(timeout=15.0) as c:
                return self._check(c.get(f"{base}/health")).json()
        return self._try(go)

    def passage(self, lang: str | None = None) -> str:
        def go(base):
            with httpx.Client(timeout=15.0) as c:
                url = f"{base}/passage" + (f"?lang={lang}" if lang else "")
                return self._check(c.get(url)).json().get("text", "")
        return self._try(go)

    def status(self) -> dict:
        def go(base):
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{base}/status")
                return r.json() if r.status_code < 400 else {}
        return self._try(go)

    def profiles(self) -> list[dict]:
        def go(base):
            with httpx.Client(timeout=30.0) as c:
                r = self._check(c.get(f"{base}/profiles", headers=self._headers()))
                return r.json().get("profiles", [])
        return self._try(go)

    def enrol(self, name: str, wav_path, text: str, notes: str = "") -> dict:
        def go(base):
            with httpx.Client(timeout=self.timeout) as c:
                with open(wav_path, "rb") as fh:
                    r = c.post(f"{base}/profiles", headers=self._headers(),
                               data={"name": name, "text": text, "notes": notes},
                               files={"file": (Path(wav_path).name, fh, "audio/wav")})
                return self._check(r).json()
        return self._try(go)

    def export_profile(self, slug: str) -> bytes:
        def go(base):
            with httpx.Client(timeout=self.timeout) as c:
                r = self._check(c.get(f"{base}/profiles/{slug}/export",
                                      headers=self._headers()))
                return r.content
        return self._try(go)

    def delete(self, slug: str) -> dict:
        def go(base):
            with httpx.Client(timeout=30.0) as c:
                return self._check(c.delete(f"{base}/profiles/{slug}",
                                            headers=self._headers())).json()
        return self._try(go)

    def speak(self, text: str, profile: str = "", takes: int = 3,
              language: str | None = None) -> dict:
        body = {"text": text, "takes": takes}
        if profile:
            body["profile"] = profile
        if language:
            body["language"] = language

        def go(base):
            with httpx.Client(timeout=self.timeout) as c:
                return self._check(
                    c.post(f"{base}/speak", json=body, headers=self._headers())
                ).json()
        out = self._try(go)
        out["audio"] = base64.b64decode(out.pop("audio_b64"))
        return out
