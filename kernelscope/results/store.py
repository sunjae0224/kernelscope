"""One parquet file per write, safe for independent concurrent writers."""
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
