"""Read-only loaders. Summary indexes keep thousands of result files cheap to browse."""
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from kernelscope.results.store import ResultStore, SCHEMA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT.parent / "kernelscope" / "results"
_LAUNCH0 = ["grid_blocks", "block_threads", "regs", "smem_bytes", "blocks_per_sm_limit", "sm_coverage",
            "occupancy_device"]


def results_root() -> Path:
    if "KERNELSCOPE_RESULTS" in os.environ:
        return Path(os.environ["KERNELSCOPE_RESULTS"]).expanduser()
    return DEFAULT_ROOT if DEFAULT_ROOT.exists() else PROJECT_ROOT / "demo_data"


def find_dirs(kind: str, root=None) -> list[Path]:
    root = Path(root) if root is not None else results_root()
    if kind == "hw":
        return sorted({p.parent for p in (root / "hw_4090").rglob("summaries.jsonl")})
    if kind == "sim":
        return sorted({p.parent for p in (root / "sim_4090").rglob("*.parquet")})
    if kind == "serve":
        roots = {root / "serve_4090", root / "serve"}
        if root.resolve() in (DEFAULT_ROOT.resolve(), (PROJECT_ROOT / "demo_data").resolve()):
            roots.add(PROJECT_ROOT / "results" / "serve")
        scenarios = set()
        for base in roots:
            for p in base.rglob("steps.parquet"):
                scenarios.add(p.parent.parent.parent if p.parent.name.startswith("repeat_") else p.parent.parent)
        return sorted(scenarios)
    raise ValueError(f"Unknown result kind: {kind}")


def load_rows(dirs) -> pd.DataFrame:
    frames = [ResultStore(d).load().assign(source_dir=str(d)) for d in dirs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    if "cache_state" not in df:
        df["cache_state"] = "warm"
    df["cache_state"] = df["cache_state"].fillna("warm")
    return df


def load_index(dirs) -> pd.DataFrame:
    """Use recorded summaries, with parquet fallback for older result directories.

    Invalid trailing JSON (a concurrent writer) is skipped. Duplicate repetitions
    are retained; the analysis layer takes their median within a workload/variant.
    """
    frames = []
    for directory in dirs:
        directory = Path(directory)
        records = []
        path = directory / "summaries.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("status") == "ok" and row.get("kernel_time_us") is not None:
                        records.append(row)
                except (ValueError, TypeError):
                    continue
        if records:
            frames.append(pd.DataFrame(records).rename(columns={"plugin": "kernel"}).assign(source_dir=str(directory)))
        else:
            rows = load_rows([directory])
            if rows.empty:
                continue
            for state in rows.cache_state.unique():
                table = kernel_table(rows, state).reset_index()
                frames.append(table.assign(cache_state=state, source_dir=str(directory)))
    if not frames:
        return pd.DataFrame(columns=["kernel", "workload_key", "cache_state", "kernel_time_us", "source_dir"])
    df = pd.concat(frames, ignore_index=True)
    if "cache_state" not in df:
        df["cache_state"] = "warm"
    df["cache_state"] = df["cache_state"].fillna("warm")
    df["kernel_time_us"] = pd.to_numeric(df.kernel_time_us, errors="coerce")
    return df[df.kernel_time_us.gt(0)].reset_index(drop=True)


def index_rows(index: pd.DataFrame) -> pd.DataFrame:
    return index.rename(columns={"kernel_time_us": "value"}).assign(
        backend="profile", metric="kernel_time_us", launch_idx=0)


def load_cell(directory, kernel, workload_key, cache_state) -> pd.DataFrame:
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{kernel}_{workload_key}")
    paths = sorted(Path(directory).glob(prefix + "*.parquet"))
    # Older writers may use an opaque tag instead of a workload prefix.
    rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else load_rows([directory])
    if rows.empty:
        return rows
    states = rows["cache_state"].fillna("warm") if "cache_state" in rows else pd.Series("warm", index=rows.index)
    return rows[(rows.kernel == kernel) & (rows.workload_key == workload_key) & (states == cache_state)].copy()


def kernel_table(rows: pd.DataFrame, cache_state: str) -> pd.DataFrame:
    states = rows["cache_state"].fillna("warm") if "cache_state" in rows else pd.Series("warm", index=rows.index)
    r = rows[states == cache_state]

    def pick(backend, metric, launch0=False):
        sel = r[(r.backend == backend) & (r.metric == metric)]
        if launch0 and "launch_idx" in sel:
            sel = sel[sel.launch_idx == 0]
        return sel.groupby(["kernel", "workload_key"])["value"].median()

    return pd.DataFrame({
        "kernel_time_us": pick("profile", "kernel_time_us"),
        "latency_us": pick("latency", "median_s") * 1e6,
        "launches": pick("profile", "launches_per_iter"),
        **{m: pick("profile", m, True) for m in _LAUNCH0},
        "achieved_gbps": pick("analytic", "achieved_gbps"), "dram_util": pick("analytic", "dram_util"),
    })


def prediction_table(workload_key, family, cache_state, machine_path, params_path, scales=None):
    """CPU-only surrogate predictions; never classified as hardware evidence."""
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.predict import DENSE_VARIANTS, PAGED_VARIANTS, predict
    from kernelscope.workload import Workload

    machine = MachineSpec.from_json(machine_path).scaled(**(scales or {}))
    params = ModelParams.from_json(params_path)
    workload = Workload.from_key(workload_key)
    variants = DENSE_VARIANTS if family == "dense" else PAGED_VARIANTS
    return pd.DataFrame([asdict(predict(v, workload, machine, params, cache_state)) for v in variants])


def load_manifest(scenario_dir) -> dict:
    path = Path(scenario_dir) / "manifest.json"
    if not path.exists():
        return {"evidence_kind": "unclassified", "performance_claim": False}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"evidence_kind": "unclassified", "performance_claim": False}
    except (OSError, ValueError):
        return {"evidence_kind": "unclassified", "performance_claim": False}


def load_serving(scenario_dir) -> dict:
    out = {}
    if not Path(scenario_dir).exists():
        return out
    for policy_dir in sorted(Path(scenario_dir).iterdir()):
        if not policy_dir.is_dir():
            continue
        runs = sorted(p.parent for p in policy_dir.glob("repeat_*/steps.parquet"))
        if (policy_dir / "steps.parquet").exists():
            runs = [policy_dir]
        if not runs:
            continue
        out[policy_dir.name] = {}
        for kind in ("steps", "tokens", "prefill"):
            frames = []
            for i, run in enumerate(runs):
                path = run / f"{kind}.parquet"
                if not path.exists():
                    continue
                try:
                    frames.append(pd.read_parquet(path).assign(repeat=i))
                except (OSError, ValueError):
                    # A campaign can be open while its next run is being saved.
                    # Incomplete files supply no evidence until the next refresh.
                    continue
            if frames:
                out[policy_dir.name][kind] = pd.concat(frames, ignore_index=True)
        if "steps" not in out[policy_dir.name]:
            del out[policy_dir.name]
    return out


def tpot_samples(tokens):
    if tokens.empty or not {"rid", "step", "t_us"}.issubset(tokens):
        return pd.Series(dtype=float)
    keys = (["repeat"] if "repeat" in tokens else []) + ["rid"]
    # The first generated token has no inter-token interval. Include its timestamp
    # as the anchor if prefill emits it; only positive intervals form TPOT samples.
    samples = tokens.sort_values(keys + ["step"]).groupby(keys)["t_us"].diff().dropna()
    return samples[samples > 0]


def amdahl_speedup(attention_fraction: float, kernel_speedup: float, overhead_fraction: float = 0.0) -> float:
    if not 0 <= attention_fraction <= 1 or not math.isfinite(kernel_speedup) or kernel_speedup <= 0:
        raise ValueError("Expected attention share in [0, 1] and positive finite kernel speedup")
    if not math.isfinite(overhead_fraction) or overhead_fraction < 0:
        raise ValueError("Dispatch overhead must be finite and non-negative")
    return 1.0 / (1.0 - attention_fraction + attention_fraction / kernel_speedup + overhead_fraction)
