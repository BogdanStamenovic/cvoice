"""Streaming: the segmenter's cut points, and both streaming routes end to end
against a stand-in engine, so it runs without a GPU or the weights."""

import json

import numpy as np
import pytest

from cvoice import server, stream
from cvoice.stream import Segmenter


def feed_tokens(text, size=3):
    seg = Segmenter()
    out = []
    for i in range(0, len(text), size):
        out += seg.feed(text[i:i + size])
    return out + seg.flush()


def test_first_chunk_is_the_first_clause_with_two_words():
    assert feed_tokens("Da, evo proveravam, sve je u redu sa serverom.")[0] == "Da, evo proveravam,"


def test_decimal_point_is_not_a_boundary():
    pieces = feed_tokens("Verzija 3.5 je izašla. Sada radi bolje nego ikad pre, stvarno.")
    assert pieces[0].startswith("Verzija 3.5 je")


def test_later_chunks_are_sentences_not_clauses():
    text = ("Dobro, evo. Pogledao sam logove, server radi normalno, samo je jedan "
            "zadatak pao noćas. Javiću ti kad završim proveru svega ostalog.")
    pieces = feed_tokens(text)
    assert pieces[0] == "Dobro, evo."
    assert pieces[1].endswith("noćas.")
    assert " ".join(pieces) == " ".join(text.split())


def test_long_run_without_marks_is_capped():
    pieces = feed_tokens("reč " * 200)
    assert all(len(p) <= stream.MAX_CHARS for p in pieces)
    assert sum(len(p.split()) for p in pieces) == 200


def test_tokenisation_does_not_change_the_chunks():
    text = "Zdravo Bogdane, evo me. Sve radi kako treba i nema grešaka. Ćao."
    assert feed_tokens(text, 1) == feed_tokens(text, 7) == stream.split(text)


def test_pcm_round_trip():
    a = np.linspace(-1, 1, 101, dtype=np.float32)
    msg = stream.chunk_message(0, "x", a, 24000, 0.1)
    assert msg["samples"] == 101
    assert np.allclose(stream.decode_chunk(msg), a, atol=1e-4)


def _speechlike(sr, seconds=0.5, floor=0.003):
    rng = np.random.default_rng(1)
    t = np.arange(int(sr * seconds)) / sr
    env = np.clip(t / 0.04, 0, 1)                 # a soft 40 ms onset
    return (env * 0.3 * np.sin(2 * np.pi * 200 * t) + rng.standard_normal(t.size) * floor).astype(np.float32)


def test_tidy_strips_only_digital_silence_and_keeps_soft_onsets():
    sr = 24000
    speech = _speechlike(sr)
    padded = np.concatenate([np.zeros(sr // 10), speech, np.zeros(sr // 10)]).astype(np.float32)
    out = stream.tidy(padded, sr, "Dobro.", last=True)
    assert out.size == speech.size
    assert np.allclose(out[:200], speech[:200])        # the onset survives untouched


def test_pauses_are_room_tone_at_the_chunk_s_own_level_never_zeros():
    sr = 24000
    speech = _speechlike(sr, floor=0.003)
    quiet = (np.random.default_rng(2).standard_normal(sr // 5) * 0.003).astype(np.float32)
    chunk = np.concatenate([speech, quiet, speech])
    for text, kind in (("Dobro, evo,", "clause"), ("Dobro, evo.", "sentence")):
        out = stream.tidy(chunk, sr, text, last=False)
        added = out.size - chunk.size
        x = int(stream.XFADE * sr)
        assert added == int(stream.PAUSE_AFTER[kind] * sr) - x
        middle = out[-int(stream.PAUSE_AFTER[kind] * sr) + x: -x]
        level = float(np.sqrt(np.mean(middle ** 2)))
        assert 0.5 * 0.003 < level < 2 * 0.003, level
        assert np.count_nonzero(middle == 0) == 0


def test_chunks_render_without_the_library_s_edge_padding(client):
    client.post("/speak/stream", json={"text": TEXT}, headers={"Authorization": "Bearer t"})
    assert client.fake.gen == {**stream.GEN, "urgent": True}


class FakeEngine:
    SAMPLE_RATE = 24000

    def __init__(self, *a, **k):
        self.calls = []

    def speak(self, text, ref_wav, ref_text, seed=None, steps=None, **gen):
        self.calls.append((text, steps))
        self.gen = gen
        return np.full(240 * len(text), 0.1, dtype=np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    prof = tmp_path / "profiles" / "test-voice"
    prof.mkdir(parents=True)
    (prof / "ref.wav").write_bytes(b"RIFF")
    (prof / "ref.txt").write_text("ref")
    fake = FakeEngine()
    monkeypatch.setattr(server, "Engine", lambda *a, **k: fake)
    monkeypatch.setattr(server.Engine, "SAMPLE_RATE", 24000, raising=False)
    cfg = {"server": {"data_dir": str(tmp_path), "token": "t"},
           "client": {"profile": "test-voice"}}
    c = TestClient(server.build_app(cfg))
    c.fake = fake
    return c


TEXT = "Dobro, evo proveravam. Pogledao sam logove i server radi normalno, sve je u redu."


def test_http_stream_sends_chunks_then_done(client):
    r = client.post("/speak/stream", json={"text": TEXT},
                    headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert [m["text"] for m in lines[:-1]] == stream.split(TEXT)
    assert lines[-1] == {"done": True, "chunks": len(lines) - 1}
    assert {s for _, s in client.fake.calls} == {stream.DEFAULT_STEPS}


def test_http_stream_needs_the_token(client):
    assert client.post("/speak/stream", json={"text": TEXT}).status_code == 401


def test_ws_turns_deltas_into_ordered_chunks(client):
    with client.websocket_connect("/speak/ws", headers={"Authorization": "Bearer t"}) as ws:
        ws.send_json({"steps": 16})
        for i in range(0, len(TEXT), 4):
            ws.send_json({"text": TEXT[i:i + 4]})
        ws.send_json({"end": True})
        got = []
        while "done" not in (m := ws.receive_json()):
            got.append(m)
    assert [m["seq"] for m in got] == list(range(len(got)))
    assert [m["text"] for m in got] == stream.split(TEXT)
    assert {s for _, s in client.fake.calls} == {16}


def test_ws_rejects_a_bad_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/speak/ws", headers={"Authorization": "Bearer nope"}) as ws:
            ws.receive_json()


def test_ws_ignores_a_query_string_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/speak/ws?token=t") as ws:
            ws.receive_json()


def test_ws_reports_an_unknown_profile(client):
    with client.websocket_connect("/speak/ws", headers={"Authorization": "Bearer t"}) as ws:
        ws.send_json({"profile": "nobody", "text": "Zdravo, kako si danas?", "end": True})
        assert "no such profile" in ws.receive_json()["error"]


def test_an_urgent_waiter_goes_before_a_batch_waiter_that_queued_first():
    import threading
    import time

    from cvoice.engine import PriorityLock

    lock = PriorityLock()
    order = []
    lock.acquire()                       # a render in flight

    def batch():
        with lock:
            order.append("batch")

    def urgent():
        with lock.urgent():
            order.append("urgent")

    b = threading.Thread(target=batch)
    b.start()
    time.sleep(0.05)                     # batch is queued first
    u = threading.Thread(target=urgent)
    u.start()
    time.sleep(0.05)
    lock.release()
    b.join(2)
    u.join(2)
    assert order == ["urgent", "batch"]
