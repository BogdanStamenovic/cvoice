"""Whisper CER/WER for every clip steps_bench.py wrote. A screen, not a verdict."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvoice.asr import Scorer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/out")
rows = json.loads((out / "results.json").read_text())
from steps_bench_lines import LINES as lines  # noqa: E402
scorer = Scorer("large-v3", "auto", "float16")
for r in rows:
    w, c, _, hyp = scorer.score([out / r["file"]], lines[r["line"]], language="sr")[0]
    r.update(wer=round(w, 1), cer=round(c, 1), hyp=hyp)
    print(r["steps"], r["line"], r["run"], r["cer"], hyp, flush=True)
(out / "scored.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
agg = defaultdict(list)
for r in rows:
    agg[r["steps"]].append(r["cer"])
for s, v in agg.items():
    print(f"steps {s}: mean CER {sum(v)/len(v):.1f}%  max {max(v):.1f}%")
