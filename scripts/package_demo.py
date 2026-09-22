"""Package recorded measurements for an offline demo; never manufacture timing rows.

Run from any directory: python scripts/package_demo.py --results /path/to/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=project.parent / "kernelscope" / "results")
    parser.add_argument("--out", type=Path, default=project / "demo_data")
    args = parser.parse_args()
    source, target = args.results.resolve(), args.out.resolve()
    if source == target or source in target.parents:
        raise SystemExit("The portable bundle must be outside the source results tree")
    files = sorted(source.glob("hw_4090/*/summaries.jsonl"))
    files += sorted(p for p in source.glob("serve_4090/**/*") if p.is_file()
                    and p.suffix in {".json", ".jsonl", ".yaml", ".parquet", ".csv"})
    if not files:
        raise SystemExit(f"No recorded measurements found in {source}")
    entries = []
    for path in files:
        rel = path.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        entries.append({"path": rel.as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema_version": 1, "kind": "recorded_measurement_bundle",
                "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source), "files": entries,
                "notes": ["Copied recorded artifacts; no synthetic performance data.",
                          "Kernel benchmark summaries are not full LLM serving measurements.",
                          "Null check_ok means correctness was not evaluated for that cell.",
                          "Raw CUDA traces and per-launch profiler parquet are omitted from the portable kernel bundle."]}
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Compact nearest-neighbour policy input derived from the recorded paged S1
    # grid. This is a measured table, not a new validation result.
    sys.path.insert(0, str(project))
    import pandas as pd
    from kernelscope.analysis.dispatch import dispatch_table
    rows = []
    for group in ("uniform_s1_paged", "ragged_s1_paged"):
        path = target / "hw_4090" / group / "summaries.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("cache_state") == "cold":
                rows.append(dict(workload_key=row["workload_key"], kernel=row["plugin"],
                                 cache_state="cold", backend="profile", metric="kernel_time_us",
                                 value=row["kernel_time_us"]))
    if rows:
        dispatch_table(pd.DataFrame(rows), family="paged", cache_state="cold").to_csv(
            target / "dispatch_paged_cold.csv", index=False)
    print(f"Packaged {len(files)} files ({sum(f['bytes'] for f in entries):,} bytes) in {target}")


if __name__ == "__main__":
    main()
