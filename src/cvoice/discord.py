# -*- coding: utf-8 -*-
"""Optional Discord delivery. Client-side on purpose: the bot token lives with
whoever runs the client, and never travels to the cvoice server."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx

API = "https://discord.com/api/v10"


def configured(cfg: dict) -> bool:
    d = cfg.get("discord", {})
    return bool(d.get("enabled")) and bool(d.get("bot_token")) and bool(d.get("channel_id"))


def post(cfg: dict, wav_path, content: str = "", filename: str | None = None) -> str:
    d = cfg["discord"]
    name = filename or Path(wav_path).name
    if d.get("mention_user_id"):
        content = f"<@{d['mention_user_id']}> {content}".strip()

    boundary = "----cvoice" + uuid.uuid4().hex
    payload = {"content": content[:1900], "attachments": [{"id": 0, "filename": name}]}
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
            f"Content-Type: application/json\r\n\r\n{json.dumps(payload)}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; "
             f"filename=\"{name}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += Path(wav_path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    r = httpx.post(
        f"{API}/channels/{d['channel_id']}/messages", content=body, timeout=120.0,
        headers={"Authorization": "Bot " + d["bot_token"],
                 "User-Agent": "cvoice/0.1",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    if r.status_code >= 400:
        raise RuntimeError(f"discord {r.status_code}: {r.text[:200]}")
    return r.json().get("id", "")
