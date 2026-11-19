"""One-off: summarize prompt-length distributions of the three public traces into a small bundle file
(demo_data/background/prompt_length_hist.json). The figure script and `kernelscope verify` read only that file.
Run: .venv/bin/python docs/presentation/background/prep_length_hist.py"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "../kernelscope/results/datasets/traces_20261024"
EDGES = np.unique(np.round(np.logspace(np.log10(16), np.log10(131072), 41)).astype(int))   # 16 .. 128K tokens
THRESH = [4096, 8192, 16384, 32768, 65536]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(name, x, path, note):
    x = np.asarray(x, dtype=np.int64)
    h, _ = np.histogram(x, bins=EDGES)
    return {"name": name, "source": str(path.name), "sha256": sha(path), "note": note, "n": int(len(x)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": int(x.max()), **{f"n_ge_{t}": int((x >= t).sum()) for t in THRESH},
            "hist_edges": EDGES.tolist(), "hist_counts": h.tolist()}


out = []
for name, f in (("Azure conv", "AzureLLMInferenceTrace_conv.csv"), ("Azure code", "AzureLLMInferenceTrace_code.csv")):
    d = pd.read_csv(D / f)
    out.append(summarize(name, d.ContextTokens, D / f, "전체 1시간"))
b = pd.read_csv(D / "BurstGPT_without_fails_1.csv")
b = b[b.Timestamp < 604800]    # first 7 days, the window the trace replays use (--window-s 604800)
b = b[b["Request tokens"] > 0]
out.append(summarize("BurstGPT (7일)", b["Request tokens"], D / "BurstGPT_without_fails_1.csv", "첫 7일, 요청 토큰 > 0"))
dst = ROOT / "demo_data/background/prompt_length_hist.json"
dst.write_text(json.dumps({"traces": out}, ensure_ascii=False, indent=1))
for o in out:
    print(o["name"], o["n"], o["p50"], o["p99"], o["max"], o["n_ge_4096"], o["n_ge_8192"], o["n_ge_32768"])
