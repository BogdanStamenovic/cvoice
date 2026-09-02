# -*- coding: utf-8 -*-
"""Configuration, read from ~/.config/cvoice/config.toml.

Every secret in this file (the API token, the Discord bot token) stays on the
machine that owns it: the client never learns the server's Discord credentials
and the server never learns the client's, because posting happens client-side.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    import tomli as tomllib

CONFIG_DIR = Path(os.environ.get("CVOICE_CONFIG_DIR", Path.home() / ".config/cvoice"))
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "client": {
        "server": "http://127.0.0.1:8760",
        "token": "",
        "profile": "",
        "takes": 3,
        "save_dir": "",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8760,
        "token": "",
        "data_dir": str(Path.home() / ".local/share/cvoice"),
        "language": "sr",
        "device": "auto",
        "model": "k2-fsa/OmniVoice",
        "asr_model": "large-v3",
    },
    "discord": {
        "enabled": False,
        "bot_token": "",
        "channel_id": "",
        "mention_user_id": "",
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in (over or {}).items():
        if isinstance(values, dict):
            out.setdefault(section, {}).update(values)
        else:
            out[section] = values
    return out


def load() -> dict:
    data = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    cfg = _merge(DEFAULTS, data)
    # Environment beats file, so a one-off override never means editing config.
    if os.environ.get("CVOICE_SERVER"):
        cfg["client"]["server"] = os.environ["CVOICE_SERVER"]
    if os.environ.get("CVOICE_TOKEN"):
        cfg["client"]["token"] = os.environ["CVOICE_TOKEN"]
    if os.environ.get("CVOICE_PROFILE"):
        cfg["client"]["profile"] = os.environ["CVOICE_PROFILE"]
    return cfg


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"%s"' % str(value).replace('"', '\\"')


def save(cfg: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# cvoice configuration", ""]
    for section in ("client", "server", "discord"):
        if section not in cfg:
            continue
        lines.append("[%s]" % section)
        for key, value in cfg[section].items():
            lines.append("%s = %s" % (key, _fmt(value)))
        lines.append("")
    CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")
    # The file holds bearer tokens; do not leave it group/world readable.
    CONFIG_PATH.chmod(0o600)
    return CONFIG_PATH


def new_token() -> str:
    return secrets.token_urlsafe(32)


def data_dir(cfg: dict) -> Path:
    p = Path(cfg["server"]["data_dir"]).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
