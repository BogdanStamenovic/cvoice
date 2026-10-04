"""/unload must release the Whisper scorer too (2026-10-04: it kept ~3.9 GiB)."""

import os
import tomllib

from cvoice import server
from cvoice.asr import Scorer


def test_scorer_unload_drops_model():
    s = Scorer()
    assert s.unload() is False  # idempotent on an idle scorer
    s._model = object()
    assert s.loaded
    assert s.unload() is True
    assert not s.loaded


def test_unload_route_releases_scorer(monkeypatch):
    from fastapi.testclient import TestClient

    made = []

    class LoadedScorer(Scorer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._model = object()
            made.append(self)

    monkeypatch.setattr(server, "Scorer", LoadedScorer)
    path = os.path.expanduser("~/.config/cvoice/config.toml")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["server"]["token"] = ""
    client = TestClient(server.build_app(cfg))
    assert client.get("/health?memory=1").json()["memory"]["scorer_loaded"] is True
    body = client.post("/unload").json()
    assert body["scorer_unloaded"] is True
    assert client.get("/health?memory=1").json()["memory"]["scorer_loaded"] is False
    assert not made[0].loaded
