"""GET /health stays the same by default and reports memory only when asked."""

from cvoice import server


def test_gpu_memory_reports_scorer_and_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr("subprocess.run", boom)
    out = server.gpu_memory(scorer_loaded=True)
    assert out["scorer_loaded"] is True
    assert "process_mib" not in out  # degraded, not failed


def test_gpu_memory_finds_own_pid(monkeypatch):
    import os
    import types

    fake = types.SimpleNamespace(stdout=f"{os.getpid()}, 1234\n99999999, 5\n")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: fake)
    assert server.gpu_memory(scorer_loaded=False)["process_mib"] == 1234
