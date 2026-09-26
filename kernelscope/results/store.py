"""One parquet file per write, safe for independent concurrent writers."""
import json
import os
import re
import uuid
from pathlib import Path

import pandas as pd

SCHEMA = ["workload_key", "kernel", "backend", "metric", "unit", "value", "launch_idx"]


class ResultStore:
    def __init__(self, root):
        self.root = Path(root)

    def write(self, rows, tag="results", extra=None):
        self.root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        for key, value in (extra or {}).items():
            frame[key] = value
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        path = self.root / f"{name}-{uuid.uuid4().hex}.parquet"
        temporary = path.with_suffix(".tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self):
        files = sorted(self.root.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=SCHEMA)
        return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def load_dirs(dirs) -> pd.DataFrame:
    """Rows from each directory's parquet files, or from its ``summaries.jsonl`` when there are none.

    The portable demo bundle (``demo_data/hw_4090``) keeps only the per-cell summaries, so those are
    turned into ``profile``/``kernel_time_us`` rows. Non-ok cells and a truncated trailing line from
    an interrupted writer are skipped.
    """
    frames = []
    for d in dirs:
        rows = ResultStore(d).load()
        path = Path(d) / "summaries.jsonl"
        if rows.empty and path.exists():
            records = []
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("status") == "ok" and r.get("kernel_time_us") is not None:
                    records.append({"workload_key": r["workload_key"], "kernel": r["plugin"], "backend": "profile",
                                    "metric": "kernel_time_us", "unit": "usecond",
                                    "value": float(r["kernel_time_us"]), "launch_idx": 0,
                                    # rows without a cache_state were measured warm (analysis/dispatch.py)
                                    "cache_state": r.get("cache_state", "warm")})
            rows = pd.DataFrame(records, columns=SCHEMA + ["cache_state"])
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
