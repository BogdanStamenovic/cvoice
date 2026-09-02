# -*- coding: utf-8 -*-
"""cvoiced - the HTTP side of cvoice.

Holds the model warm and owns the profile store. Everything is behind a bearer
token; bind to localhost or a tailnet address unless you have a reason not to.
"""
# NOTE: deliberately no `from __future__ import annotations` here. The routes
# below are defined inside build_app(), so postponed (string) annotations leave
# FastAPI unable to resolve `SpeakIn` and `UploadFile` - they are function
# locals, not module globals. That surfaced as 422 on /speak and 500 on
# /profiles. Keep annotations eager in this module.
import argparse
import base64
import os
import sys
import io
import tempfile
from pathlib import Path
from typing import Optional

from . import __version__, config
from .asr import Scorer
from .engine import Engine
from .profiles import Ambiguous, Store, slugify

PASSAGE_DIR = Path(__file__).resolve().parent.parent.parent / "passages"


def _passage(lang):
    for cand in (PASSAGE_DIR / f"{lang}.txt", PASSAGE_DIR / "sr.txt"):
        if cand.exists():
            return cand.read_text(encoding="utf-8").strip()
    return ""


def _ensure_cuda_libpath():
    """Put torch's vendored CUDA libraries on the linker path, re-execing once.

    ctranslate2 (faster-whisper) dlopens libcublas/libcudnn by soname and only
    searches the linker path. In a venv those live under site-packages/nvidia,
    which is not on it, so scoring dies with "Library libcublas.so.12 is not
    found". LD_LIBRARY_PATH is read by the loader at process start, so it cannot
    be fixed from inside - hence the re-exec. Preloading via ctypes works in a
    fresh process but not once torch has already mapped its own copies, which is
    exactly the situation in a warm server.
    """
    import glob

    if os.name != "posix" or os.environ.get("CVOICE_LIBPATH_SET"):
        return
    dirs = []
    for base in sys.path:
        if not base.endswith("site-packages"):
            continue
        dirs.extend(sorted({os.path.dirname(h) for h in
                            glob.glob(os.path.join(base, "nvidia", "*", "lib", "*.so*"))}))
    if not dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in current.split(":") for d in dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))
    os.environ["CVOICE_LIBPATH_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def build_app(cfg):
    from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel

    scfg = cfg["server"]
    root = config.data_dir(cfg)
    store = Store(root)
    engine = Engine(scfg.get("model", "k2-fsa/OmniVoice"), scfg.get("device", "auto"))
    scorer = Scorer(scfg.get("asr_model", "large-v3"), scfg.get("device", "auto"))
    token = scfg.get("token") or ""

    app = FastAPI(title="cvoice", version=__version__)
    bearer = HTTPBearer(auto_error=False)

    def auth(creds: HTTPAuthorizationCredentials = Depends(bearer)):
        if not token:
            return  # unset token means an open server; the installer warns about it
        if creds is None or creds.credentials != token:
            raise HTTPException(status_code=401, detail="bad or missing token")

    class SpeakIn(BaseModel):
        text: str
        profile: Optional[str] = None
        takes: int = 3
        language: Optional[str] = None
        seed: Optional[int] = None

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "version": __version__,
            "model_loaded": engine.loaded,
            "asr_available": scorer.available,
            "profiles": len(store.list()),
            "language": scfg.get("language", "sr"),
        }

    @app.get("/status")
    def status():
        """What a long request is actually doing. Unauthenticated on purpose:
        it carries no data, and a client that cannot read it during a first-run
        download is exactly the case this exists to fix."""
        st = dict(engine.status)
        st["model_loaded"] = engine.loaded
        return st

    @app.get("/passage")
    def passage(lang: Optional[str] = None):
        return {"text": _passage(lang or scfg.get("language", "sr"))}

    @app.get("/profiles", dependencies=[Depends(auth)])
    def list_profiles():
        return {"profiles": [
            {"slug": p["slug"], "name": p["meta"].get("name", p["slug"]),
             "duration": p["meta"].get("duration"), "notes": p["meta"].get("notes", "")}
            for p in store.list()]}

    @app.post("/profiles", dependencies=[Depends(auth)])
    async def create_profile(name: str = Form(...), text: str = Form(""),
                             notes: str = Form(""), file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            meta = store.save(name, tmp, text, notes=notes or "cvoice")
        finally:
            os.unlink(tmp)
        return {"ok": True, "profile": meta}

    @app.get("/profiles/{slug}/export", dependencies=[Depends(auth)])
    def export_profile(slug: str):
        """Return a profile as a .tar.gz so it can be moved between servers."""
        import io as _io
        import tarfile

        from fastapi.responses import Response

        try:
            resolved = store.resolve(slug)
        except Ambiguous as exc:
            raise HTTPException(409, f"'{slug}' matches: {', '.join(exc.candidates)}")
        if not resolved:
            raise HTTPException(404, f"no such profile: {slug}")
        src = store.path(resolved)
        buf = _io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(src), arcname=resolved)
        return Response(
            content=buf.getvalue(), media_type="application/gzip",
            headers={"Content-Disposition":
                     f'attachment; filename="cvoice-{resolved}.tar.gz"'})

    @app.delete("/profiles/{slug}", dependencies=[Depends(auth)])
    def delete_profile(slug: str):
        if not store.delete(slug):
            raise HTTPException(404, f"no such profile: {slug}")
        return {"ok": True}

    @app.post("/speak", dependencies=[Depends(auth)])
    def speak(req: SpeakIn):
        import soundfile as sf

        if not req.text.strip():
            raise HTTPException(400, "empty text")
        wanted = req.profile or cfg["client"].get("profile") or ""
        try:
            slug = store.resolve(wanted)
        except Ambiguous as exc:
            raise HTTPException(
                409, f"'{wanted}' matches several profiles: {', '.join(exc.candidates)}")
        if not slug:
            have = ", ".join(p["slug"] for p in store.list()) or "(none)"
            raise HTTPException(404, f"no such profile: {wanted or '(unset)'}; have: {have}")

        prof = store.load(slug)
        takes = max(1, min(int(req.takes or 1), 8))
        lang = req.language or scfg.get("language", "sr")

        tmpdir = Path(tempfile.mkdtemp(prefix="cvoice-"))
        paths = []
        try:
            for i in range(takes):
                audio = engine.speak(req.text, prof["wav"], prof["text"],
                                     seed=(req.seed + i) if req.seed is not None else 4000 + i)
                p = tmpdir / f"take{i + 1}.wav"
                sf.write(str(p), audio, Engine.SAMPLE_RATE)
                paths.append(p)

            scores = []
            best = paths[0]
            scoring_error = None
            if takes > 1 and scorer.available:
                # Scoring is a nicety; generation is the product. A broken ASR
                # stack must not turn a good clip into a 500.
                try:
                    rows = scorer.score(paths, req.text, language=lang)
                    scores = [{"wer": round(w, 1), "cer": round(c, 1),
                               "transcript": h, "file": Path(f).name}
                              for w, c, f, h in rows]
                    best = Path(rows[0][2])
                except Exception as exc:
                    scoring_error = f"{type(exc).__name__}: {exc}"

            raw = best.read_bytes()
            info = sf.info(str(best))
            return {
                "ok": True,
                "profile": {"slug": slug, "name": prof["meta"].get("name", slug)},
                "duration": round(info.duration, 2),
                "samplerate": info.samplerate,
                "takes": takes,
                "scored": bool(scores),
                "scores": scores,
                "scoring_error": scoring_error,
                "audio_b64": base64.b64encode(raw).decode(),
            }
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
            tmpdir.rmdir()

    return app


def main() -> int:
    ap = argparse.ArgumentParser(prog="cvoiced", description="cvoice server")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--preload", action="store_true",
                    help="load the model at startup instead of on first request")
    args = ap.parse_args()

    _ensure_cuda_libpath()

    cfg = config.load()
    host = args.host or cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])
    if not cfg["server"].get("token"):
        print("! no server.token set - this server accepts unauthenticated requests",
              flush=True)

    try:
        app = build_app(cfg)
    except ModuleNotFoundError as exc:
        print(f"server dependencies missing ({exc.name}). "
              f"Reinstall with the server role:\n"
              f"  python3 install.py --role server", file=sys.stderr)
        return 1
    if args.preload:
        print("preloading model...", flush=True)
        Engine(cfg["server"]["model"], cfg["server"]["device"]).load()

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
