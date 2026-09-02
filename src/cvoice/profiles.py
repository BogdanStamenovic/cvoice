# -*- coding: utf-8 -*-
"""Voice profile store.

A profile is a directory holding the reference clip, its exact transcript and
metadata. Everything that clones a voice resolves a name to one of these, so
adding a speaker never means touching the tools.

    <data_dir>/profiles/<slug>/ref.wav
    <data_dir>/profiles/<slug>/ref.txt
    <data_dir>/profiles/<slug>/meta.json
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


class Ambiguous(Exception):
    """A name matched more than one profile - a different problem from a miss."""

    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__(", ".join(candidates))


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFD", (name or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("đ", "d").replace("ђ", "d")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "profile"


class Store:
    def __init__(self, root: Path):
        self.root = Path(root) / "profiles"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, slug: str) -> Path:
        return self.root / slug

    def exists(self, slug: str) -> bool:
        return (self.path(slug) / "ref.wav").is_file()

    def load(self, slug: str) -> dict:
        p = self.path(slug)
        meta = {}
        if (p / "meta.json").exists():
            meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        text = ""
        if (p / "ref.txt").exists():
            text = (p / "ref.txt").read_text(encoding="utf-8").strip()
        return {"slug": slug, "wav": str(p / "ref.wav"), "text": text, "meta": meta}

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        return [self.load(d.name) for d in sorted(self.root.iterdir())
                if d.is_dir() and self.exists(d.name)]

    def resolve(self, token: str) -> str | None:
        """Accept a slug, a display name, or an unambiguous prefix."""
        if not token:
            return None
        token = token.strip()
        if self.exists(token):
            return token
        slug = slugify(token)
        if self.exists(slug):
            return slug
        hits = [p["slug"] for p in self.list()
                if p["slug"].startswith(slug)
                or slugify(p["meta"].get("name", "")).startswith(slug)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise Ambiguous(hits)
        return None

    def save(self, name: str, wav_src, text: str, notes: str = "") -> dict:
        import soundfile as sf

        slug = slugify(name)
        p = self.path(slug)
        p.mkdir(parents=True, exist_ok=True)
        dst = p / "ref.wav"
        if Path(wav_src).resolve() != dst.resolve():
            shutil.copyfile(wav_src, dst)
        (p / "ref.txt").write_text((text or "").strip() + "\n", encoding="utf-8")
        info = sf.info(str(dst))
        meta = {
            "name": name.strip(),
            "slug": slug,
            "duration": round(info.duration, 2),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": notes,
        }
        (p / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def delete(self, slug: str) -> bool:
        p = self.path(slug)
        if not p.is_dir():
            return False
        shutil.rmtree(p)
        return True
