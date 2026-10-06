"""Policy lab: the data behind the dashboard's experiment console (no Streamlit, no torch here).

Three comparisons read from recorded artifacts, one command composer, and a small launcher:
``protocol_comparison`` lines up the same serving scenario measured under different policy-cache
protocols; ``backend_comparison`` puts the FlashInfer kernel next to the best FlashAttention-2
variant per kernel cell; ``divergence_overview`` gathers the divergence-event summaries of recorded
campaigns; ``compose_command`` builds the exact ``serve run`` / ``bench`` argv (a grid, or one ``--workload``) the page shows, and
``launch``/``tail`` run it on a GPU host with the output in a log file.
"""
import json
import math
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from kernelscope.analysis.dispatch import FAMILIES
from kernelscope.workload import Workload, format_lens

PROTOCOL_COLUMNS = ["experiment", "campaign_group", "model", "scenario_family", "scenario_sha256", "seed", "policy_cache",
                    "policy", "repeats", "tpot_ms_mean", "speedup_vs_heuristic", "attn_ms_per_step", "step_ms_per_step",
                    "decode_wall_ms_per_step", "policy_us_per_step", "cache_misses_per_run", "tokens_equivalent"]
BACKEND_COLUMNS = ["workload_key", "B", "L_kv", "ragged", "lens", "heuristic_us", "best_fa2_kernel", "best_fa2_us",
                   "flashinfer_tensorcore_us", "flashinfer_cudacore_us", "flashinfer_kernel", "flashinfer_us",
                   "flashinfer_over_best_fa2", "heuristic_over_flashinfer", "heuristic_over_best_fa2"]
FLASHINFER = {"flashinfer_paged": "flashinfer_tensorcore_us", "flashinfer_paged_cudacore": "flashinfer_cudacore_us"}
KINDS = ("serve_run", "bench")


def _cache_misses_per_run(directory, policy) -> float:
    runs = sorted((Path(directory) / policy).glob("repeat_*/steps.parquet"))
    counts = []
    for run in runs:
        steps = pd.read_parquet(run)
        if "policy_cache_hit" not in steps or not steps.policy_cache_hit.notna().any():
            return math.nan                       # recorded before the cache flag existed, or a policy without a cache
        counts.append(int(steps.policy_cache_hit.eq(False).sum()))
    return float(sum(counts)) / len(counts) if counts else math.nan


def protocol_comparison(experiment_dirs) -> pd.DataFrame:
    """One row per (experiment, policy) with the protocol the manifest records, for side-by-side comparison."""
    from kernelscope.serve.report import summarize
    rows = []
    for directory in experiment_dirs:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        summary = summarize(directory)
        parts = directory.resolve().parts
        anchor = next((i for i, part in enumerate(parts) if part in ("serve_4090", "serve")), None)
        campaign = parts[anchor + 1] if anchor is not None and anchor + 1 < len(parts) else directory.parent.name
        scenario = manifest.get("scenario_family") or Path(str(manifest.get("scenario", directory.name))).stem
        for record in summary.to_dict("records"):
            rows.append({"experiment": str(directory), "campaign_group": campaign, "model": manifest.get("model"),
                         "scenario_family": scenario, "scenario_sha256": manifest.get("scenario_sha256"),
                         "seed": manifest.get("seed"), "policy_cache": manifest.get("policy_cache", "unrecorded"),
                         "policy": record["policy"], "repeats": record.get("repeats"),
                         **{c: record.get(c, math.nan) for c in ("tpot_ms_mean", "speedup_vs_heuristic", "attn_ms_per_step",
                                                                 "step_ms_per_step", "decode_wall_ms_per_step", "policy_us_per_step")},
                         "cache_misses_per_run": _cache_misses_per_run(directory, record["policy"]),
                         "tokens_equivalent": record.get("tokens_equivalent")})
    return pd.DataFrame(rows, columns=PROTOCOL_COLUMNS)


def backend_comparison(index: pd.DataFrame, cache_state: str = "cold") -> pd.DataFrame:
    """Per kernel cell: the library heuristic, the best FlashAttention-2 paged variant and FlashInfer (when measured).

    Both FlashInfer variants are kept; ``flashinfer_us`` is the faster one, the comparison a user of FlashInfer
    would get after picking the variant for the head shape."""
    if index.empty:
        return pd.DataFrame(columns=BACKEND_COLUMNS)
    family = FAMILIES["paged"]
    states = index["cache_state"].fillna("warm") if "cache_state" in index else pd.Series("warm", index=index.index)
    sel = index[(states == cache_state) & (index.kernel.str.match(family["members"]) | index.kernel.isin(FLASHINFER))]
    if sel.empty:
        return pd.DataFrame(columns=BACKEND_COLUMNS)
    medians = sel.groupby(["workload_key", "kernel"]).kernel_time_us.median().unstack("kernel")
    rows = []
    for key, r in medians.iterrows():
        fa2 = r.drop(labels=list(FLASHINFER), errors="ignore").dropna()
        if fa2.empty:
            continue
        w = Workload.from_key(key)
        heuristic = float(r.get(family["heuristic"], math.nan))
        best_kernel, best = fa2.idxmin(), float(fa2.min())
        variants = {name: float(r[name]) for name in FLASHINFER if name in r and pd.notna(r[name])}
        fi_kernel = min(variants, key=variants.get) if variants else None
        fi = variants[fi_kernel] if variants else math.nan
        rows.append({"workload_key": key, "B": w.B, "L_kv": w.L_kv, "ragged": w.is_ragged, "lens": format_lens(w.lens()),
                     "heuristic_us": heuristic, "best_fa2_kernel": best_kernel, "best_fa2_us": best,
                     **{column: variants.get(name, math.nan) for name, column in FLASHINFER.items()},
                     "flashinfer_kernel": fi_kernel, "flashinfer_us": fi,
                     "flashinfer_over_best_fa2": fi / best, "heuristic_over_flashinfer": heuristic / fi,
                     "heuristic_over_best_fa2": heuristic / best})
    return pd.DataFrame(rows, columns=BACKEND_COLUMNS).sort_values("heuristic_over_best_fa2", ascending=False, ignore_index=True)


def divergence_overview(campaign_dirs) -> pd.DataFrame:
    """Divergence-event summaries of recorded campaigns (see kernelscope.serve.divergence), one block per campaign."""
    from kernelscope.serve.divergence import SUMMARY_COLUMNS, campaign_events, summarize_campaign_events
    frames = []
    for directory in campaign_dirs:
        directory = Path(directory)
        try:
            events = campaign_events(directory)
        except (OSError, ValueError):
            continue
        if events.empty:
            continue
        frames.append(summarize_campaign_events(events).assign(campaign=directory.name))
    if not frames:
        return pd.DataFrame(columns=["campaign", *SUMMARY_COLUMNS])
    return pd.concat(frames, ignore_index=True)[["campaign", *SUMMARY_COLUMNS]]


def compose_command(kind, python=".venv/bin/python", model="Qwen/Qwen3-4B-Instruct-2507", scenario=None, policies=(),
                    out=None, kv_gib=10, warmup_runs=1, warmup_steps=2, repeats=3, seed=0, policy_cache="fresh",
                    machine="machines/rtx4090.json", params="models/rtx4090.json", grid=None, plugins=(), results=None,
                    cache_state="cold", workload=None, warmup=None, iters=None) -> list[str]:
    """argv of the measurement the page describes; shown verbatim so a run is reproducible
    from the shell."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    argv = [python, "-m", "kernelscope.cli"]
    if kind == "serve_run":
        argv += ["serve", "run", "--model", model, "--scenario", str(scenario)]
        for policy in policies:
            argv += ["--policy", policy]
        argv += ["--machine", machine, "--params", params, "--kv-gib", str(kv_gib), "--warmup-runs", str(warmup_runs)]
        if warmup_steps is not None:
            argv += ["--warmup-steps", str(warmup_steps)]
        argv += ["--repeats", str(repeats), "--seed", str(seed), "--policy-cache", policy_cache, "--out", str(out)]
    else:
        argv += ["bench"]
        argv += ["--workload", str(workload)] if workload is not None else ["--grid", str(grid)]
        argv += ["--plugins", ",".join(plugins), "--results", str(results), "--cache-state", cache_state]
        if warmup is not None:
            argv += ["--warmup", str(warmup)]
        if iters is not None:
            argv += ["--iters", str(iters)]
    return argv


def shell_line(argv) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def launch(argv, log_path, cwd=None) -> subprocess.Popen:
    """Start the command with stdout/stderr appended to ``log_path``; the caller keeps the Popen."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "a")
    return subprocess.Popen(list(argv), stdout=handle, stderr=subprocess.STDOUT, cwd=cwd)


def tail(log_path, lines: int = 40) -> str:
    log_path = Path(log_path)
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
