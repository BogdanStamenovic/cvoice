"""Time to first audio: Haiku 5.5 token stream -> /speak/ws, against the old path.

Old path: wait for the whole reply, then one /speak at 32 steps (what calls do
today, minus the `claude -p` overhead). New path: stream tokens into the
websocket as they arrive, and play each chunk as it comes back.

Also checks the stream never starves: playback of chunk N+1 can only start
when chunk N has finished playing AND chunk N+1 has arrived; any wait between
the two is a gap he would hear.

    ANTHROPIC_API_KEY=... .venv/bin/python bench/haiku_ws_bench.py [runs] [out_dir]
"""
import asyncio
import base64
import json
import os
import sys
import time
import tomllib
import urllib.request
import wave
from pathlib import Path

import httpx
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvoice.stream import decode_chunk  # noqa: E402

cfg = tomllib.load(open(Path.home() / ".config/cvoice/config.toml", "rb"))
HOST = f"{cfg['server']['host']}:{cfg['server']['port']}"
TOKEN = cfg["server"]["token"]
KEY = os.environ.get("ANTHROPIC_API_KEY") or \
    (Path.home() / ".config/anthropic/api_key").read_text().strip()
MODEL = os.environ.get("BENCH_MODEL", "claude-haiku-5-5")

SYSTEM = ("Ti si Bogdanov asistent na telefonskom pozivu. Odgovaraj na srpskom, "
          "latinicom, kratko i prirodno, kao u razgovoru: dve do četiri rečenice, "
          "bez nabrajanja, bez markdowna, brojeve piši rečima.")
if os.environ.get("BENCH_OPENERS") == "1":
    # The same hint hotline-ios's call voice carries: openers cvoiced has cached.
    SYSTEM += (" Odgovor počni kratkim samostalnim uvodom iz ove liste, pa tačka ili zarez: "
               "Važi, Dobro, Evo, Aha, Može, Razumem, Naravno, Samo sekund.")

PROMPTS = [
    "Kako stoji server, je l' sve radi?",
    "Objasni mi ukratko zašto je glas sporiji kad je duži tekst.",
    "Šta misliš, da li da večeras pustim duži test ili da sačekam jutro?",
]


async def haiku_tokens(prompt, client):
    # Thinking off: on by default for Haiku 5.5, it doubled output tokens (220
    # vs 111 on the same prompt, 2026-10-30) and every one of them is latency.
    body = {"model": MODEL, "max_tokens": 300, "stream": True, "system": SYSTEM,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": KEY, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    async with client.stream("POST", "https://api.anthropic.com/v1/messages",
                             json=body, headers=headers, timeout=60) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                yield ev["delta"]["text"]


async def new_path(prompt, client):
    t0 = time.perf_counter()
    marks = {}
    chunks = []
    async with websockets.connect(f"ws://{HOST}/speak/ws", max_size=2**26,
                                  additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        async def send():
            async for tok in haiku_tokens(prompt, client):
                marks.setdefault("first_token", time.perf_counter() - t0)
                await ws.send(json.dumps({"text": tok}))
            marks["last_token"] = time.perf_counter() - t0
            await ws.send(json.dumps({"end": True}))

        async def recv():
            while True:
                m = json.loads(await ws.recv())
                if "error" in m:
                    raise RuntimeError(m["error"])
                if m.get("done"):
                    return
                m["arrived"] = time.perf_counter() - t0
                chunks.append(m)

        await asyncio.gather(send(), recv())
    # Simulated playback: when would each chunk start, and how long did we wait?
    play_at, gaps = chunks[0]["arrived"], []
    for c in chunks:
        start = max(play_at, c["arrived"])
        gaps.append(round(start - play_at, 3))
        play_at = start + c["duration"]
    return {"first_token": round(marks["first_token"], 3),
            "first_audio": round(chunks[0]["arrived"], 3),
            "last_token": round(marks["last_token"], 3),
            "chunks": len(chunks), "audio_s": round(sum(c["duration"] for c in chunks), 2),
            "gaps": gaps[1:], "first_text": chunks[0]["text"],
            "gen_s": [c["gen_s"] for c in chunks],
            "text": " ".join(c["text"] for c in chunks)}, chunks


async def old_path(prompt, client):
    t0 = time.perf_counter()
    text = "".join([t async for t in haiku_tokens(prompt, client)])
    t_llm = time.perf_counter() - t0
    body = json.dumps({"text": text, "takes": 1}).encode()
    req = urllib.request.Request(f"http://{HOST}/speak", body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    d = await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(req, timeout=120)))
    return {"llm_done": round(t_llm, 3), "first_audio": round(time.perf_counter() - t0, 3),
            "audio_s": d["duration"]}


def write_wav(path, chunks):
    a = np.concatenate([decode_chunk(c) for c in chunks])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(chunks[0]["samplerate"])
        w.writeframes((a * 32767).astype("<i2").tobytes())


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "bench/out-ws")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    async with httpx.AsyncClient(http2=False) as client:
        await new_path("Reci samo: zdravo.", client)  # warm model, CUDA, TLS
        for i, prompt in enumerate(PROMPTS):
            for run in range(runs):
                new, chunks = await new_path(prompt, client)
                write_wav(out / f"p{i}-r{run}.wav", chunks)
                old = await old_path(prompt, client)
                row = {"prompt": i, "run": run, "new": new, "old": old}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    fa_new = sorted(r["new"]["first_audio"] for r in rows)
    fa_old = sorted(r["old"]["first_audio"] for r in rows)
    gaps = [g for r in rows for g in r["new"]["gaps"]]
    print(f"first audio  new: median {fa_new[len(fa_new)//2]:.2f}s  range {fa_new[0]:.2f}-{fa_new[-1]:.2f}")
    print(f"first audio  old: median {fa_old[len(fa_old)//2]:.2f}s  range {fa_old[0]:.2f}-{fa_old[-1]:.2f}")
    print(f"playback gaps: {sum(1 for g in gaps if g > 0.05)} of {len(gaps)} seams > 50 ms, worst {max(gaps or [0]):.2f}s")


asyncio.run(main())
