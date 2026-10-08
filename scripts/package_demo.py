"""Package recorded measurements for an offline demo; never manufacture timing rows.

Run from any directory: python scripts/package_demo.py --results /path/to/results
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# The paged S1 grids (uniform and ragged batches) per backend: flash-attn's split variants, and FlashInfer's
# tensor-core and CUDA-core decode kernels.
FA2_GROUPS = ("uniform_s1_paged", "ragged_s1_paged")
ANY_GROUPS = FA2_GROUPS + tuple(f"{shape}_s1_{variant}" for variant in ("flashinfer", "flashinfer_cudacore")
                                for shape in ("uniform", "ragged"))


def cold_rows(bundle: Path, groups) -> list[dict]:
    """The bundle's ok cold-cache kernel times of ``groups``, as the profile rows the dispatch analyses read."""
    rows = []
    for group in groups:
        path = bundle / "hw_4090" / group / "summaries.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("cache_state") == "cold":
                rows.append(dict(workload_key=row["workload_key"], kernel=row["plugin"],
                                 cache_state="cold", backend="profile", metric="kernel_time_us",
                                 value=row["kernel_time_us"]))
    return rows


def write_tables(project: Path, bundle: Path):
    """Compact nearest-neighbour policy inputs derived from the recorded paged S1 grids. These are measured
    tables, not new validation results."""
    sys.path.insert(0, str(project))
    import pandas as pd
    from kernelscope.analysis.dispatch import dispatch_table, static_default_losses
    rows = cold_rows(bundle, FA2_GROUPS)
    if rows:
        dispatch_table(pd.DataFrame(rows), family="paged", cache_state="cold").to_csv(
            bundle / "dispatch_paged_cold.csv", index=False)
    # Library-agnostic table: the overall best of FA2 splits and both FlashInfer variants per cell, with each
    # candidate's own time, for the cells measured on all of them.
    rows = cold_rows(bundle, ANY_GROUPS)
    if rows:
        table = static_default_losses(pd.DataFrame(rows), "cold")
        table = table[table["complete"].astype(bool)]
        if not table.empty:
            table.to_csv(bundle / "dispatch_paged_cold_any.csv", index=False)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=project.parent / "kernelscope" / "results")
    parser.add_argument("--out", type=Path, default=project / "demo_data")
    parser.add_argument("--exclude", default="*_failed_*",
                        help="glob of run directory names left out of the bundle (they stay in the results tree)")
    parser.add_argument("--tables-only", action="store_true",
                        help="copy nothing, keep provenance.json: rebuild the dispatch tables from the --out bundle")
    args = parser.parse_args()
    source, target = args.results.resolve(), args.out.resolve()
    if args.tables_only:
        write_tables(project, target)
        print(f"Rebuilt the dispatch tables in {target}")
        return
    if source == target or source in target.parents:
        raise SystemExit("The portable bundle must be outside the source results tree")
    excluded = lambda p: any(fnmatch.fnmatch(part, args.exclude) for part in p.relative_to(source).parts)
    files = sorted(p for p in source.glob("hw_4090/*/summaries.jsonl") if not excluded(p))
    files += sorted(p for p in source.glob("serve_4090/**/*") if p.is_file() and not excluded(p)
                    and p.suffix in {".json", ".jsonl", ".yaml", ".parquet", ".csv"})
    # Trace replays (scripts/traffic_replay.py): the summary only; per-step tables stay outside the repo.
    files += sorted(p for p in source.glob("traffic/*/summary.json") if not excluded(p))
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
                          "Raw CUDA traces and per-launch profiler parquet are omitted from the portable kernel bundle.",
                          "traffic/*/summary.json are surrogate-model replays of public request traces (GPU-free), not measurements."]}
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_tables(project, target)
    print(f"Packaged {len(files)} files ({sum(f['bytes'] for f in entries):,} bytes) in {target}")


if __name__ == "__main__":
    main()
