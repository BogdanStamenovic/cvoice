"""POST /yield: answer first, then exit so systemd restarts cvoiced empty."""

import copy
import threading
import time

import pytest

from cvoice import config, server
from cvoice.engine import PriorityLock


@pytest.fixture
def app_and_exits(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    exits = []
    monkeypatch.setattr(server, "exit_for_restart", lambda reason: exits.append(reason))
    monkeypatch.setattr(server, "gpu_memory", lambda scorer_loaded: {"process_mib": 1234})
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["server"].update(token="t0ken", data_dir=str(tmp_path), idle_exit_seconds=0)
    app = server.build_app(cfg)
    return app, TestClient(app), exits


AUTH = {"Authorization": "Bearer t0ken"}


def test_yield_needs_the_token(app_and_exits):
    _app, client, exits = app_and_exits
    assert client.post("/yield").status_code == 401
    assert exits == []


def test_yield_answers_then_exits(app_and_exits):
    _app, client, exits = app_and_exits
    r = client.post("/yield", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["yielding"] is True
    assert body["vram_mib_before"] == 1234
    # TestClient runs background tasks after the response, before returning.
    assert exits == ["POST /yield"]


def test_yield_409_while_a_render_holds_the_lock(app_and_exits):
    app, client, exits = app_and_exits
    engine = app.state.engine
    engine._lock.acquire()
    try:
        t = time.monotonic()
        r = client.post("/yield?wait=0.3", headers=AUTH)
        assert r.status_code == 409
        assert time.monotonic() - t < 2
        assert exits == []
    finally:
        engine._lock.release()


def test_yield_waits_for_a_render_to_finish(app_and_exits):
    app, client, exits = app_and_exits
    engine = app.state.engine
    engine._lock.acquire()
    threading.Timer(0.3, engine._lock.release).start()
    r = client.post("/yield?wait=5", headers=AUTH)
    assert r.status_code == 200
    assert exits == ["POST /yield"]


def test_yield_409_while_another_request_is_open(app_and_exits):
    app, client, exits = app_and_exits
    with app.state.idle.busy():  # e.g. a call's websocket between renders
        r = client.post("/yield?wait=0.2", headers=AUTH)
    assert r.status_code == 409
    assert r.json()["detail"]["others_active"] == 1
    assert exits == []


def test_priority_lock_timeout():
    lock = PriorityLock()
    assert lock.acquire(timeout=0.1) is True
    assert lock.acquire(timeout=0.1) is False
    lock.release()
    assert lock.acquire(timeout=0.1) is True
