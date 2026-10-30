"""Time-to-audio of /speak by step count and text length, against the live cvoiced.

Run with the cvoice venv: .venv/bin/python bench/steps_bench.py [out_dir]
Writes one wav per (steps, line, run) and a results.json next to them.
"""
import base64
import json
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

cfg = tomllib.load(open(Path.home() / ".config/cvoice/config.toml", "rb"))
BASE = f"http://{cfg['server']['host']}:{cfg['server']['port']}"
TOKEN = cfg["server"]["token"]

from steps_bench_lines import LINES  # noqa: E402  (shared with score_bench.py)
STEPS = [32, 16, 8, 6, 4]
RUNS = 3


def speak(text, steps, seed):
    body = json.dumps({"text": text, "takes": 1, "steps": steps, "seed": seed}).encode()
    req = urllib.request.Request(BASE + "/speak", body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return time.perf_counter() - t, d


out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/out")
out.mkdir(parents=True, exist_ok=True)
speak("Zagrevanje.", 8, 1)  # first request pays CUDA warm-up
rows = []
for steps in STEPS:
    for name, text in LINES.items():
        for run in range(RUNS):
            wall, d = speak(text, steps, 4000 + run)
            wav = out / f"s{steps}-{name}-{run}.wav"
            key = next(k for k in ("audio", "wav", "audio_b64") if k in d)
            wav.write_bytes(base64.b64decode(d[key]))
            rows.append({"steps": steps, "line": name, "run": run, "wall": round(wall, 3),
                         "audio_s": d.get("duration"), "file": wav.name})
            print(rows[-1], flush=True)
(out / "results.json").write_text(json.dumps(rows, indent=1))
