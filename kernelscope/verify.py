"""GPU-free verification: recompute the documented headline numbers from the committed raw data.

Each check reads recorded RTX 4090 measurements (``demo_data/`` and ``docs/experiments/``), recomputes
one number with the project's own analysis code and compares it with the value the documents state.
When a check names a document, that document must still contain the stated text, so the data and the
prose cannot drift apart silently. Nothing here needs CUDA, torch or flash-attn.

Statuses: PASS, FAIL (outside tolerance), MISSING (input file or key absent), DOC_DRIFT (the number
reproduces but the document no longer states it).
"""
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from kernelscope.analysis.dispatch import dispatch_table, regret_summary
from kernelscope.results.store import load_dirs

CAMPAIGN = "graduation_20260922"
WORST_RAGGED = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
DEMO_RAGGED = "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"
P0_DIRS = ("uniform_s1_dense", "uniform_s1_paged", "uniform_s2_dense", "uniform_s2_paged",
           "ragged_s1_dense", "ragged_s1_paged", "ragged_s2_paged")


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    expected: float
    tol: float
    unit: str
    compute: Callable[[Path, Path], float]
    doc: str | None = None
    doc_text: str | None = None


# ---- kernel measurements (demo_data/hw_4090) ------------------------------------------------------

@lru_cache(maxsize=None)
def _table(data: Path, group: str, family: str) -> pd.DataFrame:
    path = data / "hw_4090" / group
    t = dispatch_table(load_dirs([path]), family=family, cache_state="cold")
    if t.empty:
        raise FileNotFoundError(path / "summaries.jsonl")
    return t.set_index("workload_key")


def _slowdown(group, family, key):
    def f(repo, data):
        r = _table(data, group, family).loc[key]
        return r.heuristic_us / r.best_us
    return f


def _uniform_regret_pct(stat):
    def f(repo, data):
        return 100 * regret_summary(_table(data, "uniform_s1_dense", "dense").reset_index())[stat]
    return f


# ---- whole-model serving (demo_data/serve_4090) ---------------------------------------------------

def _summary(data, scenario):
    return pd.read_csv(data / "serve_4090" / CAMPAIGN / scenario / "summary.csv").set_index("policy")


def _serve(scenario, policy, column):
    return lambda repo, data: float(_summary(data, scenario).loc[policy, column])


def _token_agreement_pct(scenario, policy):
    def f(repo, data):
        e = pd.read_csv(data / "serve_4090" / CAMPAIGN / scenario / "equivalence.csv")
        return 100 * float(e.loc[e.policy == policy, "token_agreement"].iloc[0])
    return f


def _amdahl_error_pct(scenario, policy="table"):
    """Step time predicted by swapping only the attention time vs the measured step time (%)."""
    def f(repo, data):
        s = _summary(data, scenario)
        h, p = s.loc["heuristic"], s.loc[policy]
        predicted = h.step_ms_per_step - h.attn_ms_per_step + p.attn_ms_per_step
        return 100 * (predicted / p.step_ms_per_step - 1)
    return f


# ---- follow-up campaign and selection cost (docs/experiments) -------------------------------------

def _followup(repo):
    f = pd.read_csv(repo / "docs" / "experiments" / "followup" / "summary.csv")
    return f.assign(valid=f.output_validation_passed.astype(str).eq("True"))


def _followup_speedup(model, agg):
    def f(repo, data):
        d = _followup(repo)
        d = d[d.model.str.contains(model) & d.scenario_family.eq("ragged") & d.policy.eq("model")]
        return float(getattr(d.speedup, agg)())
    return f


def _followup_runs(repo, data):
    return float(_followup(repo).repeats.sum())


def _followup_valid(repo, data):
    d = _followup(repo)
    return float(d[d.policy.ne("heuristic")].valid.sum())


def _cold_decision_ms(backend):
    def f(repo, data):
        j = json.loads((repo / "docs" / "experiments" / "policy-latency.json").read_text())
        case = next(c for c in j["cases"] if c["name"] == "ragged_step0")
        return case["timing"][backend]["cold_us"]["median"] / 1e3
    return f


# ---- surrogate model validation (recomputed from demo_data/hw_4090) --------------------------------

@lru_cache(maxsize=None)
def _validation(repo: Path, data: Path) -> dict:
    from kernelscope.model.fit import prepare_rows
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.validate import report
    rows = prepare_rows(load_dirs([data / "hw_4090" / d for d in P0_DIRS]),
                        MachineSpec.from_json(repo / "machines" / "rtx4090.json"))
    if not rows:
        raise FileNotFoundError(data / "hw_4090")
    full = ModelParams.from_json(repo / "models" / "rtx4090.json")
    uniform = ModelParams.from_json(repo / "models" / "rtx4090_uniform.json")
    return {(s, c): report(rows, p, s, c) for s, p in (("V1", full), ("V5", full), ("V6", uniform))
            for c in ("cold", "warm")}


def _ape_pct(set_name, cache_state):
    return lambda repo, data: 100 * _validation(repo, data)[(set_name, cache_state)]["median_ape"]


def _regret_pct(set_name, cache_state, key):
    return lambda repo, data: 100 * _validation(repo, data)[(set_name, cache_state)]["policy"][key]


@lru_cache(maxsize=None)
def _hybrid(repo: Path, data: Path) -> dict:
    from kernelscope.model.fit import prepare_rows
    from kernelscope.model.hybrid import evaluate
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    rows = prepare_rows(load_dirs([data / "hw_4090" / d for d in P0_DIRS]),
                        MachineSpec.from_json(repo / "machines" / "rtx4090.json"))
    if not rows:
        raise FileNotFoundError(data / "hw_4090")
    full = ModelParams.from_json(repo / "models" / "rtx4090.json")
    uniform = ModelParams.from_json(repo / "models" / "rtx4090_uniform.json")
    return {(s, "cold"): evaluate(rows, uniform if s == "V6" else full, s, "cold", deltas=(0.2,))
            for s in ("V1", "V5", "V6")}


def _hybrid_max_pct(set_name, cache_state):
    return lambda repo, data: 100 * _hybrid(repo, data)[(set_name, cache_state)]["policies"]["hybrid_0.2"]["max"]


def _sim_speedup(workload, split):
    """GPGPU-Sim cycles at S=1 over cycles at the given split (reduced split-KV kernel, SM7_QV100)."""
    def f(repo, data):
        d = pd.read_csv(repo / "docs" / "experiments" / "gpgpusim-splitkv-sweep.csv")
        g = d[d.workload == workload].set_index("S").gpu_sim_cycle
        return float(g[1] / g[split])
    return f


def _sim_ratio_s1(repo, data):
    d = pd.read_csv(repo / "docs" / "experiments" / "gpgpusim-splitkv-sweep.csv")
    at = lambda w: float(d[(d.workload == w) & (d.S == 1)].gpu_sim_cycle.iloc[0])  # noqa: E731
    return at("ragged") / at("uniform")


# ---- final report v2 blanks and follow-ups (2026-10-28): Accel-Sim cells against the surrogate, GPGPU-Sim wall time,
# ---- hybrid δ sweep, no-split thresholds per GPU, Mooncake long-context trace replays ----------------------------------

FINAL2 = "docs/experiments/2026-10-28-final-report-v2.md"
EVIDENCE = "docs/experiments/2026-10-24-final-evidence.md"
ACCELSIM_CELLS = {      # cell id -> (kernel, workload key) in demo_data/accelsim/validation_20260918.csv
    "fa2_b1_l1k": ("fa2", "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"),
    "fa2_b1_l8k": ("fa2", "decode_B1_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal"),
    "fa2_b16_l1k": ("fa2", "decode_B16_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"),
    "fd_b1_l1k": ("flashdecoding", "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"),
    "fd_b1_l8k": ("flashdecoding", "decode_B1_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal"),
    "fd_b16_l1k": ("flashdecoding", "decode_B16_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"),
}
DELTA_GRID = tuple(round(0.01 * i, 2) for i in range(101))
THRESHOLD_DIRS = ("codex_validation", "ragged_s1_64k_paged", "ragged_s1_dense", "ragged_s1_paged", "ragged_s2_paged",
                  "uniform_s1_64k_paged", "uniform_s1_dense", "uniform_s1_paged", "uniform_s2_dense", "uniform_s2_paged")


def _accelsim(data: Path) -> pd.DataFrame:
    return pd.read_csv(data / "accelsim" / "validation_20260918.csv").set_index(["kernel", "workload_key"])


def _accelsim_value(cell, column):
    return lambda repo, data: float(_accelsim(data).loc[ACCELSIM_CELLS[cell], column])


@lru_cache(maxsize=None)
def _accelsim_surrogate(repo: Path, data: Path) -> dict:
    """Warm surrogate time (us) of each Accel-Sim cell and its error against the profiler kernel time (%)."""
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.predict import predict
    from kernelscope.workload import Workload
    machine = MachineSpec.from_json(repo / "machines" / "rtx4090.json")
    params = ModelParams.from_json(repo / "models" / "rtx4090.json")
    hw = _accelsim(data)
    out = {}
    for cell, (kernel, key) in ACCELSIM_CELLS.items():
        us = predict(kernel, Workload.from_key(key), machine, params, "warm").time_us
        out[cell] = {"us": us, "err_pct": 100 * abs(us / float(hw.loc[(kernel, key), "kernel_time_us"]) - 1)}
    return out


def _surrogate_us(cell):
    return lambda repo, data: _accelsim_surrogate(repo, data)[cell]["us"]


def _surrogate_err(agg):
    def f(repo, data):
        errs = pd.Series([v["err_pct"] for v in _accelsim_surrogate(repo, data).values()])
        return float(getattr(errs, agg)())
    return f


def _sim_wall_s(workload, split):
    def f(repo, data):
        d = pd.read_csv(repo / "docs" / "experiments" / "gpgpusim-splitkv-sweep.csv")
        return float(d[(d.workload == workload) & (d.S == split)].sim_wall_s.iloc[0])
    return f


@lru_cache(maxsize=None)
def _delta_sweep(repo: Path, data: Path) -> dict:
    """Leave-one-out hybrid regrets on the three cold sets for every δ in DELTA_GRID (scripts/evaluate_hybrid.py)."""
    from kernelscope.model.fit import prepare_rows
    from kernelscope.model.hybrid import evaluate
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    rows = prepare_rows(load_dirs([data / "hw_4090" / d for d in P0_DIRS]),
                        MachineSpec.from_json(repo / "machines" / "rtx4090.json"))
    if not rows:
        raise FileNotFoundError(data / "hw_4090")
    full = ModelParams.from_json(repo / "models" / "rtx4090.json")
    uniform = ModelParams.from_json(repo / "models" / "rtx4090_uniform.json")
    return {s: evaluate(rows, uniform if s == "V6" else full, s, "cold", deltas=DELTA_GRID)["policies"]
            for s in ("V1", "V5", "V6")}


def _delta_worst_pct(set_name, delta):
    return lambda repo, data: 100 * _delta_sweep(repo, data)[set_name][f"hybrid_{delta:g}"]["max"]


def _delta_points(which, edge=None):
    """δ grid points where all three cold sets stay within 15 % ("pass") or keep δ = 0.2's worst regrets ("flat")."""
    def f(repo, data):
        sweep = _delta_sweep(repo, data)
        worst = lambda d: tuple(sweep[s][f"hybrid_{d:g}"]["max"] for s in sweep)  # noqa: E731
        keep = (lambda d: max(worst(d)) <= 0.15) if which == "pass" else (lambda d: worst(d) == worst(0.2))
        points = [d for d in DELTA_GRID if keep(d)]
        return float({"low": min(points), "high": max(points), None: len(points)}[edge])
    return f


def _threshold(gpu, kv_heads):
    def f(repo, data):
        from kernelscope.analysis.thresholds import load_spec, threshold_table
        return float(threshold_table(load_spec(data / "thresholds" / "no_split_threshold.json"))[gpu][kv_heads])
    return f


@lru_cache(maxsize=None)
def _launch_agreement(data: Path) -> dict:
    from kernelscope.analysis.thresholds import launch_agreement
    return launch_agreement([data / "hw_4090" / d / "summaries.jsonl" for d in THRESHOLD_DIRS])


def _mooncake_lengths(trace, stat):
    def f(repo, data):
        d = pd.read_csv(data / "traffic" / "mooncake_lengths" / f"{trace}.csv")
        i = d.input_length
        return float({"rows": len(d), "p50": i.median(), "p90": i.quantile(0.9), "p99": i.quantile(0.99), "max": i.max(),
                      "ge_32k_pct": 100 * (i >= 32768).mean(), "ge_64k_pct": 100 * (i >= 65536).mean(),
                      "out_p50": d.output_length.median()}[stat])
    return f


# ---- GPU optional experiments of 2026-10-28 (docs/report/final_v2/남은_실험.md (a)-(c)) --------------------------

FIXED_RERUN = "fixedsplit_rerun_20261028"
TRACE_TF = "trace_e2e_numerics_20261028"
FA2_LAUNCHES = "vllm_4090/profile_20261024/fa2_launches.json"


def _rerun_mismatches(repo, data):
    return float(pd.read_csv(data / "serve_4090" / FIXED_RERUN / "ragged" / "equivalence.csv").token_mismatches.sum())


def _rerun_tpot_spread(repo, data):
    """Largest over smallest TPOT among the nine non-heuristic repeats of the re-measurement."""
    t = _p3_repeats(data, FIXED_RERUN, "ragged").drop("heuristic").tpot_ms
    return float(t.max() / t.min())


@lru_cache(maxsize=None)
def _trace_tf_events(data: Path) -> pd.DataFrame:
    """The 2026-10-24 trace e2e events classified with the 1,600-call teacher-forced rows of 2026-10-28."""
    from kernelscope.serve.divergence import campaign_events, summarize_campaign_events
    events = campaign_events(data / "serve_4090" / TRACE_RUN, numerics=data / "serve_4090" / TRACE_TF / "numerics")
    return summarize_campaign_events(events).set_index(["scenario", "policy"])


def _trace_tf(policy, column):
    return lambda repo, data: float(_trace_tf_events(data).loc[(TRACE_NAME, policy), column])


def _trace_tf_flips(policy, before=None):
    def f(repo, data):
        t = pd.read_csv(data / "serve_4090" / TRACE_TF / "numerics" / TRACE_NAME / "teacher_forced_logits.csv")
        t = t[(t.policy == policy) & ~t.argmax_equal.astype(bool)]
        return float(len(t) if before is None else (t.step < before).sum())
    return f


def _trace_tf_summary(*path):
    def f(repo, data):
        value = json.loads((data / "serve_4090" / TRACE_TF / "summary.json").read_text())
        for key in path:
            value = value[key]
        return float(value)
    return f


def _fa2_launch(run, what):
    """FA2 launches of one vLLM profile run: decode launches of the batch of 32 (grid (1, 32, 8)) and the split evidence."""
    def f(repo, data):
        r = json.loads((data / FA2_LAUNCHES).read_text())["runs"][run]
        decode = [g for g in r["groups"] if g["grid"] == [1, 32, 8]]
        return float({"decode": sum(g["launches"] for g in decode), "graph": sum(g["graph_launches"] for g in decode),
                      "split": sum(g["launches"] for g in r["groups"] if g["split"]), "combine": r["combine_launches"],
                      "dur": decode[0]["dur_us_median"]}[what])
    return f


# ---- op-class step breakdown (demo_data/serve_4090/diagnose_20260927) -----------------------------

DIAG = "diagnose_20260927"
OPB = "docs/experiments/2026-09-27-op-breakdown.md"


@lru_cache(maxsize=None)
def _diagnosis(repo: Path, data: Path, scenario: str) -> dict:
    """Recompute the op-class report from the recorded parquet files (never read diagnosis.json)."""
    from kernelscope.diagnose.report import diagnose
    from kernelscope.model.machine import MachineSpec
    return diagnose(data / "serve_4090" / DIAG / scenario, MachineSpec.from_json(repo / "machines" / "rtx4090.json"),
                    repo / "demo_data" / "dispatch_paged_cold.csv", data).summary


def _diag_value(scenario, policy, *path):
    def f(repo, data):
        node = _diagnosis(repo, data, scenario)["policies"][policy]
        for key in path:
            node = node[key]
        return float(node)
    return f


def _diag_op(scenario, policy, op_class, column, scale=1.0):
    def f(repo, data):
        ops = _diagnosis(repo, data, scenario)["policies"][policy]["ops"]
        return scale * float(next(r for r in ops if r["op_class"] == op_class)[column])
    return f


def _diag_overhead_max(repo, data):
    return max(p["timer_overhead_pct"] for s in ("ragged", "uniform", "heldout_ragged")
               for p in _diagnosis(repo, data, s)["policies"].values())


# ---- hybrid-policy generation campaign and divergence classification (demo_data/serve_4090/hybrid_20261002) ----

HYBRID_RUN = "hybrid_20261002"
HYBV = "docs/experiments/2026-10-02-hybrid-validation.md"
REQUESTED = ("ragged", "uniform", "arrivals")
INVESTIGATE = ("clear", "unclassified", "not_reproduced", "missing_tokens", "mismatches_outside_events")


@lru_cache(maxsize=None)
def _hybrid_summary(data: Path, scenario: str) -> pd.DataFrame:
    """Recompute the serving summary from the recorded parquet files (never read summary.csv)."""
    from kernelscope.serve.report import summarize
    return summarize(data / "serve_4090" / HYBRID_RUN / scenario).set_index("policy")


def _hybrid_serve(scenario, policy, column):
    return lambda repo, data: float(_hybrid_summary(data, scenario).loc[policy, column])


@lru_cache(maxsize=None)
def _recorded_splits(data: Path, scenario: str, policy: str) -> tuple:
    root = data / "serve_4090" / HYBRID_RUN / scenario / policy
    runs = sorted(root.glob("repeat_*/steps.parquet"))
    if not runs:
        raise FileNotFoundError(root)
    return tuple(tuple(int(n) for n in pd.read_parquet(p).num_splits) for p in runs)


def _steps_differing(scenarios, per_repeat=False):
    """Decode steps on which hybrid recorded another num_splits than table: the sum over scenarios and repeats,
    or the per-repeat count when every repeat agrees."""
    def f(repo, data):
        counts = [sum(a != b for a, b in zip(x, y)) for s in scenarios
                  for x, y in zip(_recorded_splits(data, s, "hybrid"), _recorded_splits(data, s, "table"))]
        if per_repeat:
            if len(set(counts)) != 1:
                raise ValueError(f"repeats disagree: {counts}")
            return float(counts[0])
        return float(sum(counts))
    return f


@lru_cache(maxsize=None)
def _campaign_events(data: Path) -> pd.DataFrame:
    """Divergence events recomputed from the token histories and the teacher-forced diagnostics (never divergence.csv)."""
    from kernelscope.serve.divergence import campaign_events
    return campaign_events(data / "serve_4090" / HYBRID_RUN)


def _divergence(columns, scenarios, agg="sum"):
    """Sum of summary columns over the (scenario, policy) rows of ``scenarios``; ``agg="same"`` requires every row
    to hold the same value and returns it."""
    def f(repo, data):
        from kernelscope.serve.divergence import summarize_campaign_events
        s = summarize_campaign_events(_campaign_events(data))
        s = s[s.scenario.isin(scenarios)]
        if s.empty:
            raise KeyError(scenarios)
        values = s[[columns] if isinstance(columns, str) else list(columns)].sum(axis=1)
        if agg == "same":
            if values.nunique() != 1:
                raise ValueError(f"rows disagree: {values.tolist()}")
            return float(values.iloc[0])
        return float(values.sum())
    return f


def _clear_margin_ulps(scenario):
    def f(repo, data):
        ev = _campaign_events(data)
        rows = ev[(ev.scenario == scenario) & (ev.tie_class == "clear")].drop_duplicates(["policy", "rid", "first_position"])
        if len(rows) != 1:
            raise ValueError(f"expected exactly one clear event in {scenario}, found {len(rows)}")
        return float(rows.margin_ulps.iloc[0])
    return f


@lru_cache(maxsize=None)
def _teacher(data: Path, scenario: str) -> pd.DataFrame:
    return pd.read_csv(data / "serve_4090" / HYBRID_RUN / "numerics" / scenario / "teacher_forced_logits.csv")


def _teacher_flips(scenarios):
    return lambda repo, data: float(sum(int((~_teacher(data, s).argmax_equal).sum()) for s in scenarios))


def _teacher_max_diff(scenario):
    return lambda repo, data: float(_teacher(data, scenario).max_abs_logit_diff.max())


def _teacher_identical_pct(scenario):
    return lambda repo, data: 100 * float((_teacher(data, scenario).max_abs_logit_diff == 0).mean())


def _cache_misses(scenarios, policy="hybrid"):
    """Policy-cache misses (cold decisions) per repeat, summed over scenarios; every repeat must agree."""
    def f(repo, data):
        per_repeat = None
        for s in scenarios:
            runs = sorted((data / "serve_4090" / HYBRID_RUN / s / policy).glob("repeat_*/steps.parquet"))
            if not runs:
                raise FileNotFoundError(data / "serve_4090" / HYBRID_RUN / s / policy)
            misses = [int((pd.read_parquet(r).policy_cache_hit == False).sum()) for r in runs]  # noqa: E712
            per_repeat = misses if per_repeat is None else [a + b for a, b in zip(per_repeat, misses)]
        if len(set(per_repeat)) != 1:
            raise ValueError(f"repeats disagree: {per_repeat}")
        return float(per_repeat[0])
    return f


def _reproducibility_pct(repo, data):
    """Largest |TPOT change| of heuristic/table between graduation_20260922 and hybrid_20261002 (same scenarios)."""
    worst = 0.0
    for s in REQUESTED:
        before = _summary(data, s)
        after = _hybrid_summary(data, s)
        for policy in ("heuristic", "table"):
            worst = max(worst, abs(100 * (after.loc[policy, "tpot_ms_mean"] / before.loc[policy, "tpot_ms_mean"] - 1)))
    return worst


def _probe(data: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data / "serve_4090" / HYBRID_RUN / "numerics" / "control_uniform_fixed8" / "kernel_probe" / name)


def _probe_kernel_err(repo, data):
    """Largest |kernel output - float32 reference| over all layers, both splits, at the clear event's step."""
    t = _probe(data, "attention_vs_fp32.csv")
    return float(t[t.variant.isin(["split1", "split8"])].max_abs_err.max())


def _probe_split_diff(repo, data):
    return float(_probe(data, "attention_vs_fp32.csv").split1_vs_split8_max.max())


def _probe_single_step_logit_diff_rid8(repo, data):
    t = _probe(data, "single_step_logits.csv")
    return float(t[t.rid == 8].max_abs_logit_diff.iloc[0])


def _probe_single_step_argmax_changes(repo, data):
    t = _probe(data, "single_step_logits.csv")
    return float((t.ref_top1 != t.cand_top1).sum())


KEEP_RUN = "hybrid_keepcache_20261006"


@lru_cache(maxsize=None)
def _keep_summary(data: Path, scenario: str) -> pd.DataFrame:
    from kernelscope.serve.report import summarize
    return summarize(data / "serve_4090" / KEEP_RUN / scenario).set_index("policy")


def _keep_serve(scenario, policy, column):
    return lambda repo, data: float(_keep_summary(data, scenario).loc[policy, column])


def _keep_cache_misses(scenario, policy):
    def f(repo, data):
        runs = sorted((data / "serve_4090" / KEEP_RUN / scenario / policy).glob("repeat_*/steps.parquet"))
        if not runs:
            raise FileNotFoundError(data / "serve_4090" / KEEP_RUN / scenario / policy)
        return float(sum(int((pd.read_parquet(r).policy_cache_hit == False).sum()) for r in runs))  # noqa: E712
    return f


def _keep_divergence(repo, data):
    from kernelscope.serve.divergence import campaign_events, summarize_campaign_events
    s = summarize_campaign_events(campaign_events(data / "serve_4090" / KEEP_RUN))
    if s.empty:
        raise FileNotFoundError(data / "serve_4090" / KEEP_RUN)
    return float(s.tie_1ulp.sum()) if float(s[list(INVESTIGATE)].sum().sum()) == 0 else float("nan")


# ---- FlashInfer vs the best FlashAttention-2 split per kernel cell (demo_data/hw_4090/*_flashinfer*) ---------------

FI_GROUPS = ("ragged_s1_paged", "ragged_s1_flashinfer", "ragged_s1_flashinfer_cudacore",
             "uniform_s1_paged", "uniform_s1_flashinfer", "uniform_s1_flashinfer_cudacore")
FI_HEADLINE = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"


@lru_cache(maxsize=None)
def _flashinfer_table(data: Path) -> pd.DataFrame:
    """Per cold cell: best FA2 paged variant, the library heuristic and both FlashInfer variants (medians of the records)."""
    from kernelscope.analysis.dispatch import FAMILIES
    from kernelscope.dashboard.data import load_index
    from kernelscope.workload import Workload
    dirs = [data / "hw_4090" / g for g in FI_GROUPS]
    missing = [d for d in dirs if not (d / "summaries.jsonl").exists()]
    if missing:
        raise FileNotFoundError(missing[0] / "summaries.jsonl")
    index = load_index(dirs)
    sel = index[(index.cache_state == "cold") & (index.kernel.str.match(FAMILIES["paged"]["members"]) | index.kernel.str.startswith("flashinfer"))]
    med = sel.groupby(["workload_key", "kernel"]).kernel_time_us.median().unstack("kernel")
    fa2 = med[[c for c in med.columns if not c.startswith("flashinfer")]]
    t = pd.DataFrame({"best_fa2": fa2.min(axis=1), "heuristic": med[FAMILIES["paged"]["heuristic"]],
                      "tensorcore": med["flashinfer_paged"], "cudacore": med["flashinfer_paged_cudacore"]}).dropna()
    t["ragged"] = [Workload.from_key(k).is_ragged for k in t.index]
    t["best_flashinfer"] = t[["tensorcore", "cudacore"]].min(axis=1)
    return t


def _fi_ratio(ragged, numerator, denominator, agg):
    def f(repo, data):
        t = _flashinfer_table(data)
        t = t[t.ragged == ragged]
        if t.empty:
            raise KeyError("ragged" if ragged else "uniform")
        ratio = t[numerator] / t[denominator]
        return float(getattr(ratio, agg)())
    return f


def _fi_cells(ragged):
    return lambda repo, data: float((_flashinfer_table(data).ragged == ragged).sum())


def _fi_headline(column):
    return lambda repo, data: float(_flashinfer_table(data).loc[FI_HEADLINE, column])


# ---- two libraries' static defaults and the 64K long side (demo_data/hw_4090, 2026-10-06) ------------------------

SCOPE = "docs/experiments/2026-10-06-problem-scope.md"
GRIDS = {"32k": ("ragged_s1_paged", "ragged_s1_flashinfer", "ragged_s1_flashinfer_cudacore",
                 "uniform_s1_paged", "uniform_s1_flashinfer", "uniform_s1_flashinfer_cudacore"),
         "64k": ("ragged_s1_64k_paged", "ragged_s1_64k_flashinfer", "ragged_s1_64k_flashinfer_cudacore",
                 "uniform_s1_64k_paged", "uniform_s1_64k_flashinfer", "uniform_s1_64k_flashinfer_cudacore")}
HEADLINE_64K = "decode_B32_Lq1_Lkv65536+512x31_Hq32_Hkv8_d128_float16_causal"


@lru_cache(maxsize=None)
def _defaults_table(data: Path, grid: str) -> pd.DataFrame:
    """Per cold cell: the best of every measured variant and each static default's loss (analysis.dispatch)."""
    from kernelscope.analysis.dispatch import static_default_losses
    dirs = [data / "hw_4090" / g for g in GRIDS[grid]]
    missing = [d for d in dirs if not (d / "summaries.jsonl").exists()]
    if missing:
        raise FileNotFoundError(missing[0] / "summaries.jsonl")
    return static_default_losses(load_dirs(dirs), "cold")


def _defaults_summary(grid, cells, default, column):
    def f(repo, data):
        from kernelscope.analysis.dispatch import static_default_summary
        s = static_default_summary(_defaults_table(data, grid)).set_index(["cells", "default"])
        return float(s.loc[(cells, default), column])
    return f


# ---- one fixed split count for every cell (demo_data/hw_4090 paged grids, 2026-10-24) -----------------------------

FIXEDSPLIT = "docs/experiments/2026-10-24-final-evidence.md"
FIXED_GROUPS = {   # group -> (grids, cache state); the FlashInfer grids are included wherever the bundle has them
    "32k_ragged": (GRIDS["32k"][:3], "cold"), "32k_uniform": (GRIDS["32k"][3:], "cold"),
    "32k_uniform_warm": (GRIDS["32k"][3:4], "warm"), "64k_ragged": (GRIDS["64k"][:3], "cold"),
    "64k_uniform": (GRIDS["64k"][3:], "cold"), "s2_ragged": (("ragged_s2_paged",), "cold"),
    "s2_uniform": (("uniform_s2_paged",), "cold")}


@lru_cache(maxsize=None)
def _fixed_table(data: Path, group: str) -> pd.DataFrame:
    """Per cell of one grid group: every fixed split's time, the FA2 heuristic and the two bests (analysis.dispatch).
    ``32k_union`` is the 32K ragged and uniform cold grids together."""
    from kernelscope.analysis.dispatch import fixed_split_table
    if group == "32k_union":
        return pd.concat([_fixed_table(data, "32k_ragged"), _fixed_table(data, "32k_uniform")], ignore_index=True)
    grids, state = FIXED_GROUPS[group]
    dirs = [data / "hw_4090" / g for g in grids]
    missing = [d for d in dirs if not (d / "summaries.jsonl").exists()]
    if missing:
        raise FileNotFoundError(missing[0] / "summaries.jsonl")
    return fixed_split_table(load_dirs(dirs), state)


def _fixed(group, policy, column, against="fa2_best"):
    def f(repo, data):
        from kernelscope.analysis.dispatch import fixed_split_summary
        s = fixed_split_summary(_fixed_table(data, group), against).set_index("policy")
        return float(s.loc[policy, column])
    return f


def _fixed_s_star(repo, data):
    from kernelscope.analysis.dispatch import best_fixed_split
    return float(best_fixed_split(_fixed_table(data, "32k_union")))


def _matched_long_cells(data: Path) -> pd.DataFrame:
    """Ragged cells both grids share: B in {26, 32, 64}, one or two long requests, short 512/1024/2048."""
    import re
    t = pd.concat([_defaults_table(data, "32k"), _defaults_table(data, "64k")], ignore_index=True)
    t = t[t.ragged.astype(bool)].copy()
    parsed = t.lens.map(lambda s: re.match(r"(\d+)(?:x(\d+))?\+", s))
    t["long"] = parsed.map(lambda m: int(m.group(1)))
    t["n_long"] = parsed.map(lambda m: int(m.group(2) or 1))
    t["fa2_loss"] = t.fa2_heuristic_us / t.fa2_best_us
    return t[t.B.isin([26, 32, 64]) & t.n_long.isin([1, 2])]


def _by_long(long, column, agg):
    def f(repo, data):
        m = _matched_long_cells(data)
        m = m[m.long == long]
        if m.empty:
            raise KeyError(long)
        return float(getattr(m[column], agg)())
    return f


def _headline_64k(column):
    return lambda repo, data: float(_defaults_table(data, "64k").set_index("workload_key").loc[HEADLINE_64K, column])


# ---- trace replays (demo_data/traffic/<run>/summary.json; scripts/traffic_replay.py, GPU-free) --------------------

def _traffic(run, *path):
    def f(repo, data):
        summary = json.loads((data / "traffic" / run / "summary.json").read_text())
        for key in path:
            summary = summary[key]
        return float(summary)
    return f


# ---- load sweep (demo_data/traffic_sweep/<run>/summary.json; scripts/traffic_replay.py --fixed-splits, 2026-10-24) -----

SWEEP = "docs/experiments/2026-10-24-final-evidence.md"


def _sweep_summary(data, run):
    return json.loads((data / "traffic_sweep" / run / "summary.json").read_text())


def _sweep(run, *path):
    def f(repo, data):
        summary = _sweep_summary(data, run)
        for key in path:
            summary = summary[key]
        return float(summary)
    return f


def _sweep_runs(data, prefix):
    return sorted(p.parent.name for p in (data / "traffic_sweep").glob(f"{prefix}*/summary.json"))


def _sweep_extreme(prefix, fn, *path):
    def f(repo, data):
        values = []
        for run in _sweep_runs(data, prefix):
            v = _sweep_summary(data, run)
            for key in path:
                v = v[key]
            values.append(float(v))
        return float(fn(values))
    return f


def _sweep_cond(run, column):
    def f(repo, data):
        table = pd.read_csv(data / "traffic_sweep" / "sweep_table.csv").set_index("run")
        return float(table.loc[run, column])
    return f


def _sweep_bin(b_lo, key):
    def f(repo, data):
        bins = json.loads((data / "traffic_sweep" / "batch_bins.json").read_text())["bins"]
        return float(next(b for b in bins if b["b_lo"] == b_lo)[key])
    return f


def _sweep_repro(run):
    """Largest absolute difference between the sweep's re-run and the 2026-10-06 bundle over every shared losses field."""
    def f(repo, data):
        old = json.loads((data / "traffic" / run / "summary.json").read_text())
        new = _sweep_summary(data, run)
        shared = [k for k, v in old["losses"].items() if isinstance(v, float) and k in new["losses"]]
        return float(max(abs(old["losses"][k] - new["losses"][k]) for k in shared) + abs(old["steps"] - new["steps"]))
    return f


# ---- 64K full-model campaign (demo_data/serve_4090/longctx_20261006/ragged_64k, 2026-10-06) -----------------------

LONGCTX_RUN = "longctx_20261006"


@lru_cache(maxsize=None)
def _longctx_summary(data: Path) -> pd.DataFrame:
    from kernelscope.serve.report import summarize
    root = data / "serve_4090" / LONGCTX_RUN / "ragged_64k"
    if not root.exists():
        raise FileNotFoundError(root)
    return summarize(root).set_index("policy")


def _longctx_serve(policy, column):
    return lambda repo, data: float(_longctx_summary(data).loc[policy, column])


def _longctx_splits(policy):
    """Distinct split counts a policy chose over its first repeat (one value = the same choice every step)."""
    def f(repo, data):
        steps = pd.read_parquet(data / "serve_4090" / LONGCTX_RUN / "ragged_64k" / policy / "repeat_000" / "steps.parquet")
        return float(steps.num_splits.nunique() * 1000 + steps.num_splits.iloc[0])
    return f


@lru_cache(maxsize=None)
def _longctx_events(data: Path) -> pd.DataFrame:
    from kernelscope.serve.divergence import campaign_events, summarize_campaign_events
    return summarize_campaign_events(campaign_events(data / "serve_4090" / LONGCTX_RUN)).set_index(["scenario", "policy"])


def _longctx_divergence(policy, column):
    return lambda repo, data: float(_longctx_events(data).loc[("ragged_64k", policy), column])


def _probe64k(data: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data / "serve_4090" / LONGCTX_RUN / "numerics" / "ragged_64k" / "kernel_probe" / name)


def _probe64k_kernel_err(repo, data):
    """Largest |kernel output - float32 reference| over all 36 layers, splits 1 and 8, at the 64K event's step."""
    t = _probe64k(data, "attention_vs_fp32.csv")
    return float(t[t.variant.isin(["split1", "split8"])].max_abs_err.max())


def _probe64k_split_diff(repo, data):
    return float(_probe64k(data, "attention_vs_fp32.csv").split1_vs_candidate_max.max())


def _probe64k_layers_over_one_ulp(repo, data):
    """Layers whose split-1 vs split-8 difference exceeds one bf16 spacing at that layer's largest output."""
    from kernelscope.serve.divergence import bf16_ulp
    t = _probe64k(data, "attention_vs_fp32.csv")
    t = t[t.variant == "split8"]
    return float(sum(d > bf16_ulp(m) + 1e-12 for d, m in zip(t.split1_vs_candidate_max, t.max_abs_out)))


def _probe64k_target(column):
    def f(repo, data):
        t = _probe64k(data, "single_step_logits.csv")
        return float(t[t.rid == 0][column].iloc[0])
    return f


def _probe64k_argmax_changes(repo, data):
    t = _probe64k(data, "single_step_logits.csv")
    return float((t.ref_top1 != t.cand_top1).sum())


def _probe64k_short_requests_dlogit(repo, data):
    t = _probe64k(data, "single_step_logits.csv")
    return float(t[t.rid != 0].max_abs_logit_diff.max())


# ---- FlashInfer inside the engine (demo_data/serve_4090/flashinfer_20261006, 2026-10-08) -----------------------------

FIENGINE_RUN = "flashinfer_20261006"
FIENGINE_BACKENDS = ("flashinfer_cudacore", "flashinfer")      # CUDA-core variant, tensor-core variant (vLLM's GQA default)
FIENGINE_SCENARIOS = {"ragged": "혼합 길이(32768×1 + 512×31)", "uniform": "균일 길이(512×32)",
                      "arrivals": "요청 도착(16384×1 + 512 요청이 step 0·8·24에 합류)"}
FIENGINE_TARGET_RID = 8


@lru_cache(maxsize=None)
def _fiengine_summary(data: Path, scenario: str) -> pd.DataFrame:
    from kernelscope.serve.report import summarize
    root = data / "serve_4090" / FIENGINE_RUN / scenario
    if not root.exists():
        raise FileNotFoundError(root)
    return summarize(root).set_index("policy")


def _fiengine_serve(scenario, policy, column):
    return lambda repo, data: float(_fiengine_summary(data, scenario).loc[policy, column])


@lru_cache(maxsize=None)
def _fiengine_event_rows(data: Path) -> pd.DataFrame:
    """Classified divergence events recomputed from the token histories and teacher-forced diagnostics (never divergence.csv)."""
    from kernelscope.serve.divergence import campaign_events
    return campaign_events(data / "serve_4090" / FIENGINE_RUN)


@lru_cache(maxsize=None)
def _fiengine_events(data: Path) -> pd.DataFrame:
    from kernelscope.serve.divergence import summarize_campaign_events
    return summarize_campaign_events(_fiengine_event_rows(data)).set_index(["scenario", "policy"])


def _fiengine_divergence(scenario, policy, column):
    return lambda repo, data: float(_fiengine_events(data).loc[(scenario, policy), column])


def _fiengine_tokens_compared(scenario, policy):
    def f(repo, data):
        rows = _fiengine_event_rows(data)
        counts = rows[(rows.scenario == scenario) & (rows.policy == policy)].tokens_compared.unique()
        if len(counts) != 1:
            raise KeyError((scenario, policy))
        return float(counts[0])
    return f


def _fiengine_cascades(scenario, policy):
    """Distinct events after which the candidate never re-joins the reference history."""
    def f(repo, data):
        rows = _fiengine_event_rows(data)
        rows = rows[(rows.scenario == scenario) & (rows.policy == policy)]
        if rows.empty:
            raise KeyError((scenario, policy))
        rows = rows[rows.events > 0]
        return float(len(rows[rows.cascade.astype(bool)].drop_duplicates(["rid", "first_position"])))
    return f


def _fiengine_table_policy_sum(column):
    """The table policy over the three scenarios: every count must be zero (token-identical to the FA2 heuristic)."""
    def f(repo, data):
        s = _fiengine_events(data)
        return float(sum(s.loc[(scenario, "table"), column] for scenario in FIENGINE_SCENARIOS))
    return f


def _fiengine_same_positions_every_repeat(repo, data):
    s = _fiengine_events(data)
    return float(all(s.loc[(scenario, policy), "same_positions_every_repeat"]
                     for scenario in FIENGINE_SCENARIOS for policy in FIENGINE_BACKENDS))


def _fiengine_needs_investigation(repo, data):
    """Unclassified + not reproduced + missing tokens + mismatches outside any event, over every FlashInfer and table run."""
    s = _fiengine_events(data).loc[[(scenario, policy) for scenario in FIENGINE_SCENARIOS
                                    for policy in ("table", *FIENGINE_BACKENDS)]]
    return float(s[["unclassified", "not_reproduced", "missing_tokens", "mismatches_outside_events"]].to_numpy().sum())


def _fiengine_clear_events(data: Path, scenarios=("ragged",)) -> pd.DataFrame:
    """The clear events of both FlashInfer policies in ``scenarios``: one distinct (rid, position), found for both."""
    rows = _fiengine_event_rows(data)
    if rows.empty:
        raise FileNotFoundError(data / "serve_4090" / FIENGINE_RUN)
    clear = rows[rows.scenario.isin(scenarios) & (rows.tie_class == "clear")]
    clear = clear.drop_duplicates(["scenario", "policy", "rid", "first_position"])
    positions = set(zip(clear.rid, clear.first_position))
    if len(positions) != 1 or set(clear.policy) != set(FIENGINE_BACKENDS) or len(clear) != len(scenarios) * len(FIENGINE_BACKENDS):
        raise ValueError(f"expected one clear position shared by both FlashInfer policies in {scenarios}, found {sorted(positions)}")
    return clear


def _fiengine_clear_position(repo, data):
    """rid * 1000 + first differing position of the ragged clear event (both FlashInfer policies agree)."""
    clear = _fiengine_clear_events(data).iloc[0]
    return float(clear.rid * 1000 + clear.first_position)


def _fiengine_clear_position_anywhere(repo, data):
    """The same number, requiring that no other scenario's clear event sits anywhere else."""
    clear = _fiengine_clear_events(data, ("ragged", "uniform")).iloc[0]
    others = _fiengine_event_rows(data)
    others = others[(others.scenario == "arrivals") & (others.tie_class == "clear")]
    if not others.empty:
        raise ValueError("the arrivals scenario holds a clear event")
    return float(clear.rid * 1000 + clear.first_position)


def _fiengine_clear_value(column, policy=None, scenario="ragged"):
    def f(repo, data):
        clear = _fiengine_clear_events(data)
        clear = clear[clear.scenario == scenario]
        if policy is not None:
            clear = clear[clear.policy == policy]
        return float(clear[column].iloc[0])
    return f


def _fiengine_clear_dlogit(scenario, policy):
    def f(repo, data):
        clear = _fiengine_clear_events(data, ("ragged", "uniform"))
        return float(clear[(clear.scenario == scenario) & (clear.policy == policy)].max_abs_logit_diff.iloc[0])
    return f


def _fiengine_clear_matches_control(repo, data):
    """1 when the FlashInfer clear event is the 2026-10-02 control's: same request, position, both tokens and margin."""
    key = ["rid", "first_position", "reference_token", "candidate_token", "margin_ulps"]
    control = _campaign_events(data)
    control = control[(control.scenario == "control_uniform_fixed8") & (control.tie_class == "clear")].drop_duplicates(key)
    ours = _fiengine_clear_events(data, ("ragged", "uniform")).drop_duplicates(key)
    if len(control) != 1 or len(ours) != 1:
        raise ValueError(f"expected one clear event on each side, found {len(control)} (control) and {len(ours)} (FlashInfer)")
    return float(all(control[k].iloc[0] == ours[k].iloc[0] for k in key))


def _fiengine_rid8_prompt_same_as_control(repo, data):
    """1 when request 8's synthetic prompt (hash and length) is the same in the three scenarios and the 10-02 control."""
    def prompt(path):
        manifest = json.loads(path.read_text())
        row = next(r for r in manifest["requests"] if r["rid"] == FIENGINE_TARGET_RID)
        return row["prompt_len"], row["prompt_sha256"]
    seen = {prompt(data / "serve_4090" / FIENGINE_RUN / s / "manifest.json") for s in FIENGINE_SCENARIOS}
    seen.add(prompt(data / "serve_4090" / HYBRID_RUN / "control_uniform_fixed8" / "manifest.json"))
    return float(len(seen) == 1)


def _fiengine_manifest_kv_gib(repo, data):
    manifests = {json.loads((data / "serve_4090" / FIENGINE_RUN / s / "manifest.json").read_text())["kv_bytes"]
                 for s in FIENGINE_SCENARIOS}
    if len(manifests) != 1:
        raise ValueError(f"scenarios disagree on the KV budget: {manifests}")
    return manifests.pop() / 2 ** 30


def _fiengine_repeats(repo, data):
    repeats = {int(_fiengine_summary(data, s).loc[p, "repeats"]) for s in FIENGINE_SCENARIOS
               for p in ("heuristic", "table", *FIENGINE_BACKENDS)}
    if len(repeats) != 1:
        raise ValueError(f"policies disagree on the repeat count: {repeats}")
    return float(repeats.pop())


# kernel probes (scripts/probe_split_kernel.py --run ragged --target-step 40 --target-rid 8 --candidate-backend <backend>)

def _fiprobe(data: Path, backend: str, name: str) -> pd.DataFrame:
    return pd.read_csv(data / "serve_4090" / FIENGINE_RUN / "numerics" / "ragged" / f"kernel_probe_{backend}" / name)


def _fiprobe_layers(data: Path, backend: str, variant=None) -> pd.DataFrame:
    t = _fiprobe(data, backend, "attention_vs_fp32.csv")
    return t[t.variant == (variant or backend)].sort_values("layer")


def _fiprobe_target(data: Path, backend: str) -> pd.Series:
    t = _fiprobe(data, backend, "single_step_logits.csv")
    return t[t.rid == FIENGINE_TARGET_RID].iloc[0]


def _fiprobe_layer_count(backend):
    return lambda repo, data: float(len(_fiprobe_layers(data, backend)))


def _fiprobe_kernel_err(backend, variant=None):
    return lambda repo, data: float(_fiprobe_layers(data, backend, variant).max_abs_err.max())


def _fiprobe_split_diff(backend):
    return lambda repo, data: float(_fiprobe_layers(data, backend).split1_vs_candidate_max.max())


def _fiprobe_worst_layer(backend):
    def f(repo, data):
        t = _fiprobe_layers(data, backend)
        return float(t.loc[t.split1_vs_candidate_max.idxmax(), "layer"])
    return f


def _fiprobe_worst_layer_out(backend):
    def f(repo, data):
        t = _fiprobe_layers(data, backend)
        return float(t.loc[t.split1_vs_candidate_max.idxmax(), "max_abs_out"])
    return f


def _fiprobe_layers_over_one_ulp(backend):
    """Layers whose backend-vs-split-1 difference exceeds one bf16 spacing at that layer's largest output."""
    def f(repo, data):
        from kernelscope.serve.divergence import bf16_ulp
        t = _fiprobe_layers(data, backend)
        return float(sum(d > bf16_ulp(m) + 1e-12 for d, m in zip(t.split1_vs_candidate_max, t.max_abs_out)))
    return f


def _fiprobe_requests(backend):
    return lambda repo, data: float(len(_fiprobe(data, backend, "single_step_logits.csv")))


def _fiprobe_target_value(backend, column):
    return lambda repo, data: float(_fiprobe_target(data, backend)[column])


def _fiprobe_argmax_changes(backend):
    def f(repo, data):
        t = _fiprobe(data, backend, "single_step_logits.csv")
        return float((t.ref_top1 != t.cand_top1).sum())
    return f


def _fiprobe_batch_dlogit(backend, agg):
    return lambda repo, data: float(getattr(_fiprobe(data, backend, "single_step_logits.csv").max_abs_logit_diff, agg)())


# What the note's §4 tables print. The cells ARE the document text: a check's ``doc_text`` is the table row up to the
# cell it checks, so a changed cell in the note (or in the data) is flagged as DOC_DRIFT (or FAIL), never silently kept.

def _fiengine_row_text(cells, upto):
    return "| " + " | ".join(cells[: upto + 1]) + " |"


# scenario -> policy -> (TPOT ms, speedup vs heuristic, attention ms/step, attention share, policy us/step)
# A FlashInfer policy's policy_us is its per-step plan() cost; the heuristic is the baseline (no speedup cell).
FIENGINE_PERF = {
    "ragged": {
        "heuristic": (60.99, None, 36.78, 0.719, 3.7),
        "table": (33.70, 1.810, 9.23, 0.385, 6.7),
        "flashinfer_cudacore": (33.28, 1.833, 8.69, 0.368, 370.8),
        "flashinfer": (48.51, 1.257, 23.87, 0.616, 432.4),
    },
    "uniform": {
        "heuristic": (27.45, None, 3.58, 0.199, 3.4),
        "table": (27.43, 1.001, 3.58, 0.199, 6.5),
        "flashinfer_cudacore": (27.66, 0.992, 3.47, 0.190, 291.0),
        "flashinfer": (27.57, 0.996, 3.40, 0.187, 337.7),
    },
    "arrivals": {
        "heuristic": (59.18, None, 10.58, 0.430, 3.0),
        "table": (50.46, 1.173, 5.04, 0.263, 7.4),
        "flashinfer_cudacore": (50.12, 1.181, 4.78, 0.250, 284.2),
        "flashinfer": (54.18, 1.092, 7.09, 0.331, 315.5),
    },
}
FIENGINE_PERF_COLUMNS = (   # id suffix, summarize() column, title, tolerance, unit
    ("tpot", "tpot_ms_mean", "TPOT", 0.005, "ms"),
    ("speedup", "speedup_vs_heuristic", "TPOT 개선 배율(휴리스틱 대비)", 0.0005, "x"),
    ("attn", "attn_ms_per_step", "step당 attention 시간", 0.005, "ms"),
    ("attn_share", "attn_share", "decode step에서 attention이 차지하는 비중", 0.0005, ""),
    ("policy_us", "policy_us_per_step", "step당 정책 비용(FlashInfer 정책은 plan())", 0.05, "us"),
)


def _fiengine_perf_cells(policy, values):
    tpot, speedup, attn, share, policy_us = values
    return [policy, f"{tpot:.2f}", "기준" if speedup is None else f"{speedup:.3f}배", f"{attn:.2f}", f"{share:.1%}", f"{policy_us:.1f}"]


# (scenario, FlashInfer policy) -> (token mismatches per repeat (max), tokens compared, distinct event positions,
#                                   tie_1ulp, tie_2ulp, clear, events after which the histories never re-join)
FIENGINE_EVENTS = {
    ("ragged", "flashinfer_cudacore"): (65, 2048, 4, 3, 0, 1, 1),
    ("ragged", "flashinfer"): (92, 2048, 4, 3, 0, 1, 0),
    ("uniform", "flashinfer_cudacore"): (223, 2048, 7, 4, 2, 1, 2),
    ("uniform", "flashinfer"): (92, 2048, 4, 3, 0, 1, 0),
    ("arrivals", "flashinfer_cudacore"): (39, 1192, 6, 6, 0, 0, 1),
    ("arrivals", "flashinfer"): (53, 1192, 7, 7, 0, 0, 0),
}
FIENGINE_EVENT_COLUMNS = (  # id suffix, title, compute(scenario, policy)
    ("mismatches", "반복당 불일치 토큰 수(최대)", lambda s, p: _fiengine_divergence(s, p, "token_mismatches_max")),
    ("tokens", "비교한 토큰 수", _fiengine_tokens_compared),
    ("events", "분기 사건 수(서로 다른 위치)", lambda s, p: _fiengine_divergence(s, p, "distinct_positions")),
    ("tie_1ulp", "tie_1ulp 사건 수", lambda s, p: _fiengine_divergence(s, p, "tie_1ulp")),
    ("tie_2ulp", "tie_2ulp 사건 수", lambda s, p: _fiengine_divergence(s, p, "tie_2ulp")),
    ("clear", "clear 사건 수", lambda s, p: _fiengine_divergence(s, p, "clear")),
    ("cascade", "끝까지 갈라지는(합류하지 않는) 사건 수", _fiengine_cascades),
)


def _fiengine_event_cells(scenario, policy, values):
    mismatches, compared, *counts = values
    return [f"{scenario} · {policy}", f"{mismatches}", f"{compared:,}", *map(str, counts)]


# Kernel probes at the ragged clear event's step, same KV state: (id suffix, row label, format, tolerance,
# (CUDA-core, tensor-core) values, factory(backend) -> compute)
FIENGINE_PROBE_ROWS = (
    ("layers", "탐침한 층 수", "{:.0f}", 0, (36, 36), _fiprobe_layer_count),
    ("kernel_err", "attention 출력의 fp32 참조 대비 최대 오차", "{:.3f}", 0.0005, (0.167, 0.120), _fiprobe_kernel_err),
    ("split1_err", "FA2 분할 1의 같은 오차", "{:.3f}", 0.0005, (0.134, 0.134), lambda b: _fiprobe_kernel_err(b, "split1")),
    ("split_diff", "분할 1과의 층별 최대 차이", "{:.2f}", 0.0005, (0.25, 0.25), _fiprobe_split_diff),
    ("worst_layer", "그 차이가 최대인 층", "{:.0f}", 0, (34, 34), _fiprobe_worst_layer),
    ("worst_layer_out", "그 층의 최대 출력", "{:.2f}", 0.005, (33.13, 33.13), _fiprobe_worst_layer_out),
    ("layers_over_one_ulp", "차이가 그 층 출력의 bf16 간격 1개를 넘는 층", "{:.0f}", 0, (0, 0), _fiprobe_layers_over_one_ulp),
    ("requests", "단일 step에서 비교한 요청 수", "{:.0f}", 0, (32, 32), _fiprobe_requests),
    ("target_margin", "요청 8의 참조 1·2위 로짓 차이", "{:.3f}", 0.0005, (11.875, 11.875), lambda b: _fiprobe_target_value(b, "ref_margin")),
    ("target_top1", "요청 8의 argmax 토큰", "{:.0f}", 0, (25, 25), lambda b: _fiprobe_target_value(b, "cand_top1")),
    ("argmax_changes", "argmax가 바뀐 요청 수", "{:.0f}", 0, (0, 0), _fiprobe_argmax_changes),
    ("target_dlogit", "요청 8의 로짓 최대 차이", "{:.4f}", 0.0005, (0.3125, 1.9688),
     lambda b: _fiprobe_target_value(b, "max_abs_logit_diff")),
    ("target_cosine", "요청 8의 로짓 코사인", "{:.4f}", 0.0005, (0.9998, 0.9903), lambda b: _fiprobe_target_value(b, "cosine")),
    ("batch_dlogit_max", "배치 전체의 로짓 최대 차이", "{:.3f}", 0.0005, (2.625, 2.969), lambda b: _fiprobe_batch_dlogit(b, "max")),
    ("batch_dlogit_median", "요청별 로짓 최대 차이의 중앙값", "{:.3f}", 0.0005, (0.117, 0.125),
     lambda b: _fiprobe_batch_dlogit(b, "median")),
)


def _fiengine_probe_cells(label, fmt, values):
    return [label, *(fmt.format(v) for v in values)]


def _fiengine_generated_checks():
    out = []
    for scenario, rows in FIENGINE_PERF.items():
        for policy, values in rows.items():
            cells = _fiengine_perf_cells(policy, values)
            for i, (suffix, column, title, tol, unit) in enumerate(FIENGINE_PERF_COLUMNS):
                if values[i] is not None:
                    out.append(Check(f"fiengine.{scenario}_{policy}_{suffix}",
                                     f"엔진 안 FlashInfer, {FIENGINE_SCENARIOS[scenario]}, {policy} 정책: {title}",
                                     values[i], tol, unit, _fiengine_serve(scenario, policy, column), SCOPE,
                                     _fiengine_row_text(cells, i + 1)))
    for (scenario, policy), values in FIENGINE_EVENTS.items():
        cells = _fiengine_event_cells(scenario, policy, values)
        for i, (suffix, title, compute) in enumerate(FIENGINE_EVENT_COLUMNS):
            out.append(Check(f"fiengine.{scenario}_{policy}_{suffix}",
                             f"엔진 안 FlashInfer, {FIENGINE_SCENARIOS[scenario]}, {policy} 정책 vs 휴리스틱: {title}",
                             values[i], 0, "", compute(scenario, policy), SCOPE, _fiengine_row_text(cells, i + 1)))
    for suffix, label, fmt, tol, values, factory in FIENGINE_PROBE_ROWS:
        cells = _fiengine_probe_cells(label, fmt, values)
        for i, backend in enumerate(FIENGINE_BACKENDS):
            out.append(Check(f"fiengine.probe_{backend}_{suffix}",
                             f"FlashInfer 커널 탐침(혼합 길이, 요청 8 위치 41의 step, 같은 KV 상태), {backend}: {label}",
                             values[i], tol, "", factory(backend), SCOPE, _fiengine_row_text(cells, i + 1)))
    return out


# ---- Library-agnostic table campaign (demo_data/serve_4090/anytable_20261008, 2026-10-08) ------------------------

ANYTABLE_RUN = "anytable_20261008"
ANYTABLE_CSV = "dispatch_paged_cold_any.csv"
ANYTABLE_POLICIES = ("table", "table_any", "table_any_p371")     # FA2-only table, kernel-time-only table, table + plan() cost
ANYTABLE_BACKENDS = ("fa2", "flashinfer", "flashinfer_cudacore")
ANYTABLE_FLASHINFER_KERNELS = ("flashinfer_paged", "flashinfer_paged_cudacore")
ANYTABLE_PLAN_US = 371.0


@lru_cache(maxsize=None)
def _anytable_cells(repo: Path) -> pd.DataFrame:
    path = repo / "demo_data" / ANYTABLE_CSV
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _anytable_cell_count(repo, data):
    return float(len(_anytable_cells(repo)))


def _anytable_best_count(kernels, ragged_only=False):
    """Cells whose overall best kernel is in ``kernels`` (``None`` = any FA2 variant)."""
    def f(repo, data):
        t = _anytable_cells(repo)
        if ragged_only:
            t = t[t.ragged.astype(bool)]
        hit = ~t.best_kernel.isin(ANYTABLE_FLASHINFER_KERNELS) if kernels is None else t.best_kernel.isin(kernels)
        return float(hit.sum())
    return f


def _anytable_ragged_cells(repo, data):
    return float(_anytable_cells(repo).ragged.astype(bool).sum())


def _anytable_cudacore_gain(agg):
    """CUDA-core time saved per call against the best FA2 split, over the ragged cells where CUDA-core is the overall best."""
    def f(repo, data):
        t = _anytable_cells(repo)
        t = t[t.ragged.astype(bool) & (t.best_kernel == "flashinfer_paged_cudacore")]
        return float(getattr(t.fa2_best_us - t.flashinfer_cc_us, agg)())
    return f


def _anytable_table_is_the_grid_table(repo, data):
    """1 when the packaged table is the complete cells of ``static_default_losses`` over the six 32K grids."""
    import numpy as np
    fresh = _defaults_table(data, "32k")
    fresh = fresh[fresh.complete.astype(bool)].sort_values("workload_key").reset_index(drop=True)
    packed = _anytable_cells(repo).sort_values("workload_key").reset_index(drop=True)
    same = (list(fresh.workload_key) == list(packed.workload_key) and (fresh.best_kernel == packed.best_kernel).all()
            and (fresh.fa2_best_kernel == packed.fa2_best_kernel).all())
    for column in ("best_us", "fa2_heuristic_us", "flashinfer_tc_us", "flashinfer_cc_us", "fa2_best_us"):
        same = same and np.allclose(fresh[column], packed[column], rtol=1e-9, atol=0)
    return float(same)


def _anytable_manifest(data: Path, scenario: str = "ragged") -> dict:
    path = data / "serve_4090" / ANYTABLE_RUN / scenario / "manifest.json"
    return json.loads(path.read_text())


def _anytable_n_layers(repo, data):
    return float(_anytable_manifest(data)["model_config"]["n_layers"])


def _anytable_plan_us_default(repo, data):
    import inspect
    from kernelscope.serve.dispatch import TablePolicy
    return float(inspect.signature(TablePolicy.__init__).parameters["plan_us"].default)


def _anytable_plan_us_spec(repo, data):
    """The plan_us in the recorded ``table_any:<csv>:<plan_us>`` policy spec."""
    specs = _anytable_manifest(data)["policy_specs"]
    spec = next(s for s in specs if s.startswith("table_any:") and s.count(":") == 2)
    return float(spec.rpartition(":")[2])


@lru_cache(maxsize=None)
def _anytable_policies(repo: Path, data: Path) -> dict:
    """The three table policies as the campaign built them (n_layers from the recorded model config)."""
    from kernelscope.serve.dispatch import TablePolicy
    new, old = repo / "demo_data" / ANYTABLE_CSV, repo / "demo_data" / "dispatch_paged_cold.csv"
    for path in (new, old):
        if not path.exists():
            raise FileNotFoundError(path)
    policies = {"table": TablePolicy(old),
                "table_any": TablePolicy(new, ANYTABLE_BACKENDS, 0.0, "table_any"),
                "table_any_p371": TablePolicy(new, ANYTABLE_BACKENDS, ANYTABLE_PLAN_US, "table_any_p371")}
    for policy in policies.values():
        policy.n_layers = int(_anytable_manifest(data)["model_config"]["n_layers"])
    return policies


@lru_cache(maxsize=None)
def _anytable_steps(data: Path, scenario: str, policy: str) -> tuple:
    """steps.parquet of every recorded repeat (read-only: the frames are cached)."""
    root = data / "serve_4090" / ANYTABLE_RUN / scenario / policy
    runs = sorted(root.glob("repeat_*/steps.parquet"))
    if not runs:
        raise FileNotFoundError(root)
    return tuple(pd.read_parquet(p) for p in runs)


def _anytable_agreed(values, what):
    if len(set(values)) != 1:
        raise KeyError(f"repeats disagree on {what}: {values}")
    return float(values[0])


def _anytable_backend_steps(scenario, policy, backend):
    """Decode steps per repeat that ran ``backend`` (every repeat must agree)."""
    return lambda repo, data: _anytable_agreed(
        [int((s.attention == backend).sum()) for s in _anytable_steps(data, scenario, policy)], (scenario, policy, backend))


def _anytable_split_steps(scenario, policy, splits):
    """FA2 decode steps per repeat that passed ``num_splits == splits`` (every repeat must agree)."""
    return lambda repo, data: _anytable_agreed(
        [int(((s.attention == "fa2") & (s.num_splits == splits)).sum()) for s in _anytable_steps(data, scenario, policy)],
        (scenario, policy, splits))


def _anytable_steps_per_repeat(repo, data):
    counts = {len(s) for scenario in FIENGINE_SCENARIOS for policy in ("heuristic", *ANYTABLE_POLICIES)
              for s in _anytable_steps(data, scenario, policy)}
    return _anytable_agreed(sorted(counts), "decode steps per repeat")


def _anytable_p371_choice_differences(repo, data):
    """Steps (all scenarios and repeats) where table_any_p371 recorded another (backend, num_splits) than table."""
    return float(sum(int(((a.attention != b.attention) | (a.num_splits != b.num_splits)).sum())
                     for scenario in FIENGINE_SCENARIOS
                     for a, b in zip(_anytable_steps(data, scenario, "table_any_p371"), _anytable_steps(data, scenario, "table"))))


def _anytable_replay_differences(repo, data):
    """Steps (all scenarios, policies and repeats) whose recorded (backend, num_splits) the policy does not return for
    the recorded batch lengths when re-run on the CPU."""
    cfg = _anytable_manifest(data)["model_config"]
    wrong = 0
    for name, policy in _anytable_policies(repo, data).items():
        for scenario in FIENGINE_SCENARIOS:
            for steps in _anytable_steps(data, scenario, name):
                for lens, backend, splits in zip(steps.lens, steps.attention, steps.num_splits):
                    chosen = policy.choose(json.loads(lens), cfg["n_heads"], cfg["n_kv_heads"])
                    wrong += (policy.attention, chosen) != (backend, int(splits))
    return float(wrong)


def _anytable_cell(repo, data, scenario) -> pd.Series:
    """The table row the table_any policy read at every decode step of ``scenario`` (exactly one cell)."""
    cfg, policy = _anytable_manifest(data)["model_config"], _anytable_policies(repo, data)["table_any"]
    keys = set()
    for steps in _anytable_steps(data, scenario, "table_any"):
        for lens in steps.lens:
            policy.choose(json.loads(lens), cfg["n_heads"], cfg["n_kv_heads"])
            keys.add(policy.last["workload_key"])
    if len(keys) != 1:
        raise KeyError(f"{scenario}: table_any read {len(keys)} cells")
    cells = _anytable_cells(repo).set_index("workload_key")
    return cells.loc[keys.pop()]


def _anytable_cell_value(scenario, column):
    return lambda repo, data: float(_anytable_cell(repo, data, scenario)[column])


def _anytable_cell_fa2_splits(scenario):
    from kernelscope.model.geometry import parse_variant
    return lambda repo, data: float(parse_variant(_anytable_cell(repo, data, scenario).fa2_best_kernel).num_splits)


def _anytable_cell_gain_us(per_step):
    """CUDA-core time saved against the best FA2 split in the ragged scenario's cell: per call, or per step (all layers)."""
    def f(repo, data):
        cell = _anytable_cell(repo, data, "ragged")
        gain = float(cell.fa2_best_us - cell.flashinfer_cc_us)
        return gain * _anytable_n_layers(repo, data) if per_step else gain
    return f


def _anytable_cell_cost_gap_us(repo, data):
    """Cost of the CUDA-core pick (layers x kernel time + plan_us) minus the FA2 pick's, in the ragged cell."""
    cell, layers = _anytable_cell(repo, data, "ragged"), _anytable_n_layers(repo, data)
    return layers * (cell.flashinfer_cc_us - cell.fa2_best_us) + ANYTABLE_PLAN_US


@lru_cache(maxsize=None)
def _anytable_summary(data: Path, scenario: str) -> pd.DataFrame:
    from kernelscope.serve.report import summarize
    root = data / "serve_4090" / ANYTABLE_RUN / scenario
    if not root.exists():
        raise FileNotFoundError(root)
    return summarize(root).set_index("policy")


def _anytable_serve(scenario, policy, column):
    return lambda repo, data: float(_anytable_summary(data, scenario).loc[policy, column])


def _anytable_fiengine_serve(scenario, policy, column):
    return lambda repo, data: float(_fiengine_summary(data, scenario).loc[policy, column])


def _anytable_gap(scenario, a, b, column, scale=1.0):
    """summarize() column of policy ``a`` minus policy ``b`` (x scale)."""
    return lambda repo, data: scale * float(_anytable_summary(data, scenario).loc[a, column]
                                            - _anytable_summary(data, scenario).loc[b, column])


def _anytable_attn_gain_minus_plan_ms(scenario):
    """Attention time saved by table_any against table (ms/step) minus table_any's policy cost (ms/step)."""
    def f(repo, data):
        s = _anytable_summary(data, scenario)
        return float(s.loc["table", "attn_ms_per_step"] - s.loc["table_any", "attn_ms_per_step"]
                     - s.loc["table_any", "policy_us_per_step"] / 1e3)
    return f


def _anytable_attn_gain_per_layer_us(scenario):
    def f(repo, data):
        s = _anytable_summary(data, scenario)
        return float(s.loc["table", "attn_ms_per_step"] - s.loc["table_any", "attn_ms_per_step"]) * 1e3 / _anytable_n_layers(repo, data)
    return f


def _anytable_tpot_pct(scenario, slower, faster):
    """How much longer ``slower``'s TPOT is than ``faster``'s (%)."""
    return lambda repo, data: 100 * float(_anytable_summary(data, scenario).loc[slower, "tpot_ms_mean"]
                                          / _anytable_summary(data, scenario).loc[faster, "tpot_ms_mean"] - 1)


def _anytable_spread_pct(scenario):
    """Longest over shortest TPOT among the three table policies (%)."""
    def f(repo, data):
        tpot = _anytable_summary(data, scenario).loc[list(ANYTABLE_POLICIES), "tpot_ms_mean"]
        return 100 * float(tpot.max() / tpot.min() - 1)
    return f


@lru_cache(maxsize=None)
def _anytable_repeat_tpot(data: Path, scenario: str) -> pd.DataFrame:
    """Per-repeat TPOT (ms), one column per policy: the mean over requests of the per-request mean token gap."""
    from kernelscope.serve.report import iter_runs, tpot_us
    root = data / "serve_4090" / ANYTABLE_RUN / scenario
    rows = [(policy, repeat, float((tpot_us(pd.read_parquet(directory / "tokens.parquet")).dropna() / 1000).mean()))
            for policy, repeat, directory, _ in iter_runs(root)]
    if not rows:
        raise FileNotFoundError(root)
    return pd.DataFrame(rows, columns=["policy", "repeat", "tpot"]).pivot(index="repeat", columns="policy", values="tpot")


def _anytable_faster_repeats(scenario, faster="table_any", slower="table"):
    """Repeats in which ``faster`` had the shorter TPOT."""
    def f(repo, data):
        t = _anytable_repeat_tpot(data, scenario)
        return float((t[faster] < t[slower]).sum())
    return f


@lru_cache(maxsize=None)
def _anytable_event_rows(data: Path) -> pd.DataFrame:
    """Classified divergence events recomputed from the token histories and teacher-forced diagnostics (never divergence.csv)."""
    from kernelscope.serve.divergence import campaign_events
    return campaign_events(data / "serve_4090" / ANYTABLE_RUN)


@lru_cache(maxsize=None)
def _anytable_events(data: Path) -> pd.DataFrame:
    from kernelscope.serve.divergence import summarize_campaign_events
    return summarize_campaign_events(_anytable_event_rows(data)).set_index(["scenario", "policy"])


def _anytable_divergence(scenario, policy, column):
    return lambda repo, data: float(_anytable_events(data).loc[(scenario, policy), column])


def _anytable_tokens_compared(scenario, policy):
    def f(repo, data):
        rows = _anytable_event_rows(data)
        counts = rows[(rows.scenario == scenario) & (rows.policy == policy)].tokens_compared.unique()
        if len(counts) != 1:
            raise KeyError((scenario, policy))
        return float(counts[0])
    return f


def _anytable_cascades(scenario, policy):
    """Distinct events after which the candidate never re-joins the reference history."""
    def f(repo, data):
        rows = _anytable_event_rows(data)
        rows = rows[(rows.scenario == scenario) & (rows.policy == policy)]
        if rows.empty:
            raise KeyError((scenario, policy))
        rows = rows[rows.events > 0]
        return float(len(rows[rows.cascade.astype(bool)].drop_duplicates(["rid", "first_position"])))
    return f


def _anytable_zero_policies(column):
    """The two policies that stay on FA2 (table, table_any_p371) over the three scenarios: every count must be zero."""
    def f(repo, data):
        s = _anytable_events(data)
        return float(sum(s.loc[(scenario, policy), column] for scenario in FIENGINE_SCENARIOS
                         for policy in ("table", "table_any_p371")))
    return f


def _anytable_same_positions_every_repeat(repo, data):
    s = _anytable_events(data)
    return float(all(s.loc[(scenario, policy), "same_positions_every_repeat"]
                     for scenario in FIENGINE_SCENARIOS for policy in ANYTABLE_POLICIES))


def _anytable_needs_investigation(repo, data):
    """Unclassified + not reproduced + missing tokens + mismatches outside any event, over every table policy run."""
    s = _anytable_events(data)
    return float(s[["unclassified", "not_reproduced", "missing_tokens", "mismatches_outside_events"]].to_numpy().sum())


def _anytable_token_differences(scenario, policy, other_run, other_policy):
    """Tokens (over all repeats) that differ or exist on one side only between this campaign's ``policy`` and the
    ``other_run`` campaign's ``other_policy`` (same scenario, same repeat numbers)."""
    def f(repo, data):
        total = 0
        runs = sorted((data / "serve_4090" / ANYTABLE_RUN / scenario / policy).glob("repeat_*/tokens.parquet"))
        if not runs:
            raise FileNotFoundError(data / "serve_4090" / ANYTABLE_RUN / scenario / policy)
        for path in runs:
            theirs = data / "serve_4090" / other_run / scenario / other_policy / path.parent.name / "tokens.parquet"
            if not theirs.exists():
                raise FileNotFoundError(theirs)
            both = pd.read_parquet(path).merge(pd.read_parquet(theirs), on=["rid", "position"], how="outer", suffixes=("_a", "_b"))
            total += int(((both.token_a != both.token_b) | both.token_a.isna() | both.token_b.isna()).sum())
        return float(total)
    return f


def _anytable_p371_token_differences(repo, data):
    """The same count between table_any_p371 and table, summed over the three scenarios."""
    return float(sum(_anytable_token_differences(s, "table_any_p371", ANYTABLE_RUN, "table")(repo, data) for s in FIENGINE_SCENARIOS))


def _anytable_events_equal_fiengine(repo, data):
    """1 when, in the ragged and arrivals scenarios, table_any's divergence events (request, position, both tokens,
    class, margin, logit difference) are exactly flashinfer_cudacore's events in the 2026-10-08 FlashInfer campaign."""
    columns = ["rid", "first_position", "reference_token", "candidate_token", "tie_class", "margin_ulps", "max_abs_logit_diff"]
    ours, theirs = _anytable_event_rows(data), _fiengine_event_rows(data)
    for scenario in ("ragged", "arrivals"):
        a = ours[(ours.scenario == scenario) & (ours.policy == "table_any") & (ours.events > 0)]
        b = theirs[(theirs.scenario == scenario) & (theirs.policy == "flashinfer_cudacore") & (theirs.events > 0)]
        if a.empty or sorted(a.repeat.unique()) != sorted(b.repeat.unique()):
            raise KeyError(scenario)
        for repeat in a.repeat.unique():
            if not a[a.repeat == repeat][columns].reset_index(drop=True).equals(b[b.repeat == repeat][columns].reset_index(drop=True)):
                return 0.0
    return 1.0


def _anytable_clear_event(data: Path) -> pd.Series:
    rows = _anytable_event_rows(data)
    clear = rows[(rows.scenario == "ragged") & (rows.policy == "table_any") & (rows.tie_class == "clear")]
    clear = clear.drop_duplicates(["rid", "first_position"])
    if len(clear) != 1:
        raise KeyError(f"expected one clear event in ragged/table_any, found {len(clear)}")
    return clear.iloc[0]


def _anytable_clear_position(repo, data):
    clear = _anytable_clear_event(data)
    return float(clear.rid * 1000 + clear.first_position)


def _anytable_clear_value(column):
    return lambda repo, data: float(_anytable_clear_event(data)[column])


def _anytable_clear_is_the_fiengine_event(repo, data):
    """1 when that clear event is §4b's: same request, position, both tokens, margin and logit difference."""
    key = ["rid", "first_position", "reference_token", "candidate_token", "margin_ulps", "max_abs_logit_diff"]
    ours = _anytable_clear_event(data)
    theirs = _fiengine_clear_events(data)
    theirs = theirs[(theirs.scenario == "ragged") & (theirs.policy == "flashinfer_cudacore")].drop_duplicates(key[:2])
    if len(theirs) != 1:
        raise KeyError("expected one clear event of flashinfer_cudacore in the FlashInfer campaign")
    return float(all(ours[k] == theirs[k].iloc[0] for k in key))


def _anytable_repeats(repo, data):
    repeats = {int(_anytable_summary(data, s).loc[p, "repeats"]) for s in FIENGINE_SCENARIOS for p in ("heuristic", *ANYTABLE_POLICIES)}
    if len(repeats) != 1:
        raise KeyError(f"policies disagree on the repeat count: {repeats}")
    return float(repeats.pop())


def _anytable_kv_gib(repo, data):
    sizes = {_anytable_manifest(data, s)["kv_bytes"] for s in FIENGINE_SCENARIOS}
    if len(sizes) != 1:
        raise KeyError(f"scenarios disagree on the KV budget: {sizes}")
    return sizes.pop() / 2 ** 30


def _anytable_run_time(edge):
    """Start (first manifest created_at) or end (last completed_at) of the three runs in KST, as hour * 100 + minute."""
    from datetime import datetime, timedelta, timezone
    key = "created_at" if edge == "start" else "completed_at"

    def f(repo, data):
        stamps = [datetime.fromisoformat(_anytable_manifest(data, s)[key]) for s in FIENGINE_SCENARIOS]
        t = (min(stamps) if edge == "start" else max(stamps)).astimezone(timezone(timedelta(hours=9)))
        return float(t.hour * 100 + t.minute)
    return f


def _anytable_code_state(repo, data):
    """1 when every manifest records commit b8e1707 with uncommitted changes on top."""
    git = [_anytable_manifest(data, s)["git"] for s in FIENGINE_SCENARIOS]
    return float(all(g["commit"].startswith("b8e1707") and g["status"].strip() for g in git))


# What the note's §5 tables print (cells ARE the document text, as in the fiengine block above).

# scenario -> policy -> (TPOT ms, speedup vs heuristic, attention ms/step, attention share, policy us/step)
ANYTABLE_PERF = {
    "ragged": {
        "heuristic": (60.90, None, 36.81, 0.720, 3.9),
        "table": (33.64, 1.810, 9.23, 0.386, 7.5),
        "table_any": (33.26, 1.831, 8.69, 0.369, 382.2),
        "table_any_p371": (33.71, 1.806, 9.23, 0.386, 7.1),
    },
    "uniform": {
        "heuristic": (27.44, None, 3.58, 0.199, 3.6),
        "table": (27.40, 1.001, 3.58, 0.199, 6.7),
        "table_any": (27.42, 1.001, 3.58, 0.199, 6.6),
        "table_any_p371": (27.42, 1.001, 3.58, 0.199, 6.8),
    },
    "arrivals": {
        "heuristic": (59.08, None, 10.56, 0.431, 2.9),
        "table": (50.32, 1.174, 5.04, 0.263, 7.8),
        "table_any": (50.14, 1.178, 4.79, 0.250, 291.8),
        "table_any_p371": (50.42, 1.172, 5.04, 0.263, 7.8),
    },
}
ANYTABLE_PERF_COLUMNS = FIENGINE_PERF_COLUMNS

# (scenario, policy) -> steps per repeat on FlashInfer CUDA-core, on FA2, and the FA2 steps by num_splits 0 / 8 / 16
ANYTABLE_CHOICES = {
    ("ragged", "table"): (0, 63, 0, 0, 63),
    ("ragged", "table_any"): (63, 0, 0, 0, 0),
    ("ragged", "table_any_p371"): (0, 63, 0, 0, 63),
    ("uniform", "table"): (0, 63, 63, 0, 0),
    ("uniform", "table_any"): (0, 63, 63, 0, 0),
    ("uniform", "table_any_p371"): (0, 63, 63, 0, 0),
    ("arrivals", "table"): (0, 63, 0, 41, 22),
    ("arrivals", "table_any"): (63, 0, 0, 0, 0),
    ("arrivals", "table_any_p371"): (0, 63, 0, 41, 22),
}
ANYTABLE_CHOICE_COLUMNS = (   # id suffix, title, compute(scenario, policy)
    ("cudacore_steps", "FlashInfer CUDA-core로 돈 step 수(반복당)", lambda s, p: _anytable_backend_steps(s, p, "flashinfer_cudacore")),
    ("fa2_steps", "FA2로 돈 step 수(반복당)", lambda s, p: _anytable_backend_steps(s, p, "fa2")),
    ("fa2_split0_steps", "FA2 분할 0(FA2 휴리스틱)으로 돈 step 수", lambda s, p: _anytable_split_steps(s, p, 0)),
    ("fa2_split8_steps", "FA2 분할 8로 돈 step 수", lambda s, p: _anytable_split_steps(s, p, 8)),
    ("fa2_split16_steps", "FA2 분할 16으로 돈 step 수", lambda s, p: _anytable_split_steps(s, p, 16)),
)

# table_any vs the heuristic: (token mismatches per repeat (max), tokens compared, distinct event positions,
#                              tie_1ulp, tie_2ulp, clear, events after which the histories never re-join)
ANYTABLE_EVENTS = {
    "ragged": (65, 2048, 4, 3, 0, 1, 1),
    "uniform": (0, 2048, 0, 0, 0, 0, 0),
    "arrivals": (39, 1192, 6, 6, 0, 0, 1),
}


def _anytable_choice_cells(scenario, policy, values):
    return [f"{scenario} · {policy}", *map(str, values)]


def _anytable_event_cells(scenario, values):
    mismatches, compared, *counts = values
    return [f"{scenario} · table_any", f"{mismatches}", f"{compared:,}", *map(str, counts)]


def _anytable_generated_checks():
    out = []
    for scenario, rows in ANYTABLE_PERF.items():
        for policy, values in rows.items():
            cells = _fiengine_perf_cells(policy, values)
            for i, (suffix, column, title, tol, unit) in enumerate(ANYTABLE_PERF_COLUMNS):
                if values[i] is not None:
                    out.append(Check(f"anytable.{scenario}_{policy}_{suffix}",
                                     f"라이브러리 무관 표, {FIENGINE_SCENARIOS[scenario]}, {policy} 정책: {title}",
                                     values[i], tol, unit, _anytable_serve(scenario, policy, column), SCOPE,
                                     _fiengine_row_text(cells, i + 1)))
    for (scenario, policy), values in ANYTABLE_CHOICES.items():
        cells = _anytable_choice_cells(scenario, policy, values)
        for i, (suffix, title, compute) in enumerate(ANYTABLE_CHOICE_COLUMNS):
            out.append(Check(f"anytable.{scenario}_{policy}_{suffix}",
                             f"라이브러리 무관 표, {FIENGINE_SCENARIOS[scenario]}, {policy} 정책의 step별 선택: {title}",
                             values[i], 0, "", compute(scenario, policy), SCOPE, _fiengine_row_text(cells, i + 1)))
    for scenario, values in ANYTABLE_EVENTS.items():
        cells = _anytable_event_cells(scenario, values)
        for i, (suffix, title, _) in enumerate(FIENGINE_EVENT_COLUMNS):
            compute = {"tokens": _anytable_tokens_compared, "cascade": _anytable_cascades}.get(suffix)
            column = {"mismatches": "token_mismatches_max", "events": "distinct_positions"}.get(suffix, suffix)
            out.append(Check(f"anytable.{scenario}_table_any_{suffix}",
                             f"라이브러리 무관 표, {FIENGINE_SCENARIOS[scenario]}, table_any 정책 vs 휴리스틱: {title}",
                             values[i], 0, "", compute(scenario, "table_any") if compute else
                             _anytable_divergence(scenario, "table_any", column), SCOPE, _fiengine_row_text(cells, i + 1)))
    return out


# ---- vLLM reproduction (docs/experiments/vllm/*.json; separate vLLM environment, measurement only) ---------------

def _vllm_rows(repo: Path, run: str) -> pd.DataFrame:
    rows = json.loads((repo / "docs" / "experiments" / "vllm" / f"{run}_20261006.json").read_text())["rows"]
    return pd.DataFrame(rows).set_index(["backend", "mode", "scenario"])


def _vllm_step_ms(run, backend, mode, scenario):
    return lambda repo, data: float(_vllm_rows(repo, run).loc[(backend, mode, scenario), "decode_ms_per_step_median"])


def _vllm_ratio(run, numerator, denominator):
    def f(repo, data):
        t = _vllm_rows(repo, run).decode_ms_per_step_median
        return float(t.loc[numerator] / t.loc[denominator])
    return f


P0 = "docs/plan/2026-09-19-p0-campaign.md"
HYB = "docs/experiments/2026-09-26-hybrid-policy.md"
SIM = "docs/experiments/2026-09-26-gpgpusim-splitkv.md"
VAL = "docs/plan/2026-09-22-model-validation.md"

# ---- trace-driven scenario, vLLM backend selection, long-context and GSM8K scenarios (2026-10-24, P1b) ----------

TRACE_E2E = "docs/experiments/2026-10-24-final-evidence.md"
TRACE_SCENARIO = "scenarios/trace_azure_conv_i550_n300.yaml"
LONGTEXT_SCENARIO = "scenarios/demo_text_ragged_64k.yaml"
GSM8K_SCENARIO = "scenarios/gsm8k_ragged32k_n200.yaml"
TRACE_STEP_S = 0.03


@lru_cache(maxsize=None)
def _trace_e2e_replay(repo: Path):
    """The scenario replayed on the engine's logical-step grid (arrival_step x 30 ms) and scored by the surrogate."""
    from kernelscope.analysis.traffic import ReplayConfig, replay, step_losses
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.serve.scenarios import load
    requests = load(repo / TRACE_SCENARIO)
    trace = pd.DataFrame({"arrival_s": [r.arrival_step * TRACE_STEP_S for r in requests],
                          "prompt_tokens": [r.prompt_len for r in requests], "output_tokens": [r.max_new_tokens for r in requests]})
    steps = replay(trace, ReplayConfig(max_batch=64, kv_tokens=72_817, step_s=TRACE_STEP_S))
    losses = step_losses(steps, MachineSpec.from_json(repo / "machines/rtx4090.json"), ModelParams.from_json(repo / "models/rtx4090.json"))
    return steps, losses, requests


def _trace_e2e(what):
    def f(repo, data):
        from kernelscope.analysis.traffic import summarize_losses
        steps, losses, requests = _trace_e2e_replay(Path(repo))
        if what == "requests":
            return float(len(requests))
        if what == "steps":
            return float(len(steps))
        if what == "batch_mean":
            return float(steps.B.mean())
        if what == "criterion_steps":
            return float(((losses.loss >= 1.25) & (losses.B >= 26)).sum())
        if what == "prompt_max":
            return float(max(r.prompt_len for r in requests))
        if what == "prompts_ge_4096":
            return float(sum(r.prompt_len >= 4096 for r in requests))
        if what == "bundle_matches_recompute":      # the bundled prediction is what the scenario replays to now
            predicted = json.loads((data / "trace_e2e" / "predicted.json").read_text())
            s = summarize_losses(losses)
            same = (predicted["steps"] == len(steps) and abs(predicted["losses"]["attention_time_ratio"] - s["attention_time_ratio"]) < 1e-9)
            return float(same)
        return float(summarize_losses(losses)[what])
    return f


def _trace_selection(what):
    def f(repo, data):
        sel = json.loads((data / "trace_e2e" / "selection.json").read_text())
        scored = [c for c in sel["candidates"] if c.get("criterion_steps") is not None]
        return float({"windows": len(sel["candidates"]), "windows_reaching_batch": len(scored),
                      "chosen_start": sel["chosen"]["start"], "chosen_criterion": sel["chosen"]["criterion_steps"],
                      "relaxed": sel["criteria"]["relaxed_no_window_reached_min_batch"]}[what])
    return f


def _trace_predicted(*path):
    def f(repo, data):
        value = json.loads((data / "trace_e2e" / "predicted.json").read_text())
        for key in path:
            value = value[key]
        return float(value)
    return f


def _vllm_backend(what):
    def f(repo, data):
        d = json.loads((data / "vllm_4090" / "backend_selection.json").read_text())
        return float({"starts": d["engine_starts_reporting_backend"], "flash_attn": d["using"].get("FLASH_ATTN", 0),
                      "flashinfer": d["using"].get("FLASHINFER", 0), "warnings": d["unknown_env_var_warnings"],
                      "requested_flashinfer": d["child_runs_requesting_flashinfer"]}[what])
    return f


def _scenario_stat(path, what):
    def f(repo, data):
        from kernelscope.serve.scenarios import load_scenario
        requests, _ = load_scenario(Path(repo) / path)
        text = [r for r in requests if r.prompt_text is not None]
        longs = [r for r in requests if r.prompt_recipe == "corpus_repeat"]
        shorts = [r for r in longs if r.prompt_len < 4096]
        if what == "questions":
            return float(len(text))
        if what == "groups":
            return float(len({r.arrival_step for r in requests}))
        if what == "first_group_questions":
            return float(sum(r.arrival_step == 0 for r in text))
        if what == "max_new":
            return float(max(r.max_new_tokens for r in requests))
        if what == "spacing":
            return float(sorted({r.arrival_step for r in requests})[1])
        if what == "long_len":
            return float(max(r.prompt_len for r in longs))
        if what == "short_count":
            return float(len(shorts))
        if what == "short_len":
            return float(shorts[0].prompt_len)
        raise KeyError(what)
    return f


# ---- GPU queue results of 2026-10-24/25 (P3): fixed split in the engine, trace-replay scoring, trace-driven end to end,
# ---- vLLM step decomposition, 64K natural text, GSM8K ------------------------------------------------------------

FIXED_RUN = "fixedsplit_20261024"
TRACE_RUN = "trace_e2e_20261024"
TRACE_NAME = "trace_azure_conv_i550_n300"
LONGTEXT_RUN = "longctx_text_20261024"
GSM8K_RUN = "gsm8k_20261024"
VALIDATION_DIR = "hw_4090/trace_validation_20261024"
VLLM_PROFILE = "vllm_4090/profile_20261024"


@lru_cache(maxsize=None)
def _p3_summary(data: Path, run: str, scenario: str) -> pd.DataFrame:
    from kernelscope.serve.report import summarize
    return summarize(data / "serve_4090" / run / scenario).set_index("policy")


def _p3_serve(run, scenario, policy, column):
    return lambda repo, data: float(_p3_summary(data, run, scenario).loc[policy, column])


@lru_cache(maxsize=None)
def _p3_repeats(data: Path, run: str, scenario: str) -> pd.DataFrame:
    """Per policy and repeat: TPOT mean, policy cost and cache-hit rate, attention per step (from the recorded parquet files)."""
    from kernelscope.serve.report import iter_runs, tpot_us
    rows = []
    for policy, repeat, directory, _ in iter_runs(data / "serve_4090" / run / scenario):
        steps, tokens = pd.read_parquet(directory / "steps.parquet"), pd.read_parquet(directory / "tokens.parquet")
        rows.append(dict(policy=policy, repeat=repeat, tpot_ms=float(tpot_us(tokens).mean()) / 1000, attn_ms=float(steps.attn_us.mean()) / 1000,
                         step_ms=float(steps.step_us.mean()) / 1000,
                         policy_us=float(steps.policy_us.mean()),
                         hit=float(steps.policy_cache_hit.mean()) if "policy_cache_hit" in steps else math.nan))
    return pd.DataFrame(rows).set_index(["policy", "repeat"])


def _p3_repeat(run, scenario, policy, repeat, column):
    return lambda repo, data: float(_p3_repeats(data, run, scenario).loc[(policy, repeat), column])


def _p3_steady_tpot(run, scenario, policy):
    """Mean TPOT over the repeats after the first (decision cache filled by the first repeat)."""
    def f(repo, data):
        t = _p3_repeats(data, run, scenario).loc[policy]
        return float(t.tpot_ms.iloc[1:].mean())
    return f


def _p3_steady_speedup(run, scenario, policy):
    def f(repo, data):
        t = _p3_repeats(data, run, scenario)
        return float(t.loc["heuristic"].tpot_ms.iloc[1:].mean() / t.loc[policy].tpot_ms.iloc[1:].mean())
    return f


@lru_cache(maxsize=None)
def _p3_events(data: Path, run: str) -> pd.DataFrame:
    from kernelscope.serve.divergence import campaign_events, summarize_campaign_events
    return summarize_campaign_events(campaign_events(data / "serve_4090" / run)).set_index(["scenario", "policy"])


def _p3_event(run, scenario, policy, column):
    return lambda repo, data: float(_p3_events(data, run).loc[(scenario, policy), column])


@lru_cache(maxsize=None)
def _p3_step_attention(data: Path, run: str, scenario: str) -> pd.DataFrame:
    """Per decode step attention time averaged over the repeats, one column per policy (schedules are identical across policies)."""
    from kernelscope.serve.report import iter_runs
    frames = {}
    for policy, _, directory, _ in iter_runs(data / "serve_4090" / run / scenario):
        frames.setdefault(policy, []).append(pd.read_parquet(directory / "steps.parquet").attn_us.to_numpy())
    out = pd.DataFrame({p: sum(v) / len(v) for p, v in frames.items()})
    return out


def _p3_step_ratio(policy, what):
    """Measured per-step attention: heuristic over the policy, as a total ratio or as the fraction of steps at or above 1.25."""
    def f(repo, data):
        a = _p3_step_attention(data, TRACE_RUN, TRACE_NAME)
        ratio = a.heuristic / a[policy]
        return float(a.heuristic.sum() / a[policy].sum()) if what == "total" else float((ratio >= 1.25).mean())
    return f


def _p3_teacher(run, scenario, column):
    def f(repo, data):
        t = pd.read_csv(data / "serve_4090" / run / "numerics" / scenario / "teacher_forced_logits.csv")
        t = t[t.policy == "table"] if column != "rows_all" else t
        return {"rows": float(len(t)), "flips": float((~t.argmax_equal.astype(bool)).sum()), "rows_all": float(len(t)),
                "max_diff": float(t.max_abs_logit_diff.max()), "tied": float(t.reference_tied_top1.astype(bool).sum())}[column]
    return f


@lru_cache(maxsize=None)
def _p3_validation(data: Path, step_weight: str = "stratum"):
    from kernelscope.analysis import validation
    first = pd.read_csv(data / VALIDATION_DIR / "measured_pass1.csv")
    second = pd.read_csv(data / VALIDATION_DIR / "measured_pass2.csv")
    return validation.prepare(first, step_weight), validation.prepare(second, step_weight), first, second


def _p3_val(source, key, passes=1, step_weight="stratum"):
    def f(repo, data):
        from kernelscope.analysis import validation
        m = _p3_validation(data, step_weight)[passes - 1]
        return float(validation.source_stats(m, source)[key])
    return f


def _p3_val_ci(source, key, side):
    def f(repo, data):
        from kernelscope.analysis import validation
        return float(validation.bootstrap(_p3_validation(data)[0], source, key)[side])
    return f


def _p3_val_error(key):
    def f(repo, data):
        from kernelscope.analysis import validation
        return float(validation.overall_error(_p3_validation(data)[0])[key])
    return f


def _p3_val_stratum(source, stratum, column):
    def f(repo, data):
        from kernelscope.analysis import validation
        return float(validation.error_by_stratum(_p3_validation(data)[0], source).loc[stratum, column])
    return f


def _p3_val_noise(key):
    def f(repo, data):
        from kernelscope.analysis import validation
        _, _, first, second = _p3_validation(data)
        return float(validation.run_to_run(first, second)[key])
    return f


def _p3_val_false_positive_range(side):
    """Measured regret range of the conv cells predicted >= 1.25 whose measured regret is below it."""
    def f(repo, data):
        m = _p3_validation(data)[0]
        g = m[(m.source == "azure_conv_x1") & (m.pred_r >= 1.25) & (m.meas_r < 1.25)].meas_r
        return float(g.min() if side == "min" else g.max())
    return f


@lru_cache(maxsize=None)
def _p3_vllm(data: Path) -> dict:
    from kernelscope.analysis import vllm_steps
    runs = vllm_steps.load(data / VLLM_PROFILE)
    return {"table": vllm_steps.run_table(runs), "runs": {r["run"]: r for r in runs}}


def _p3_vllm_value(run, column):
    return lambda repo, data: float(_p3_vllm(data)["table"][run][column])


def _p3_vllm_excess_share(ragged, uniform):
    """Share of the ragged-over-uniform step time that is attention GPU time."""
    def f(repo, data):
        t = _p3_vllm(data)["table"]
        return float((t[ragged]["attention_ms"] - t[uniform]["attention_ms"]) / (t[ragged]["unprofiled_ms"] - t[uniform]["unprofiled_ms"]))
    return f


def _p3_vllm_after_leave(column):
    """The decode iteration after the 32K request finished (31 requests left), FA2 eager, in milliseconds."""
    def f(repo, data):
        from kernelscope.analysis import vllm_steps
        return float(vllm_steps.last_row_after_long_request_leaves(_p3_vllm(data)["runs"]["flash_attn_eager_ragged"])[column]) / 1000
    return f


def _p3_vllm_logged_median(run, column):
    return lambda repo, data: float(_p3_vllm(data)["runs"][run]["summary"][column]) / 1000


def _p3_vllm_grid_ratio(repo, data):
    """vLLM FA2 attention per layer / the kernel grid's FA2-heuristic time of the same batch (B=32, 32768 + 512 x 31)."""
    grid = pd.read_csv(data / "dispatch_paged_cold.csv").set_index("lens")
    layers = 36
    per_layer_ms = _p3_vllm(data)["table"]["flash_attn_eager_ragged"]["attention_ms"] / layers
    return float(per_layer_ms * 1000 / grid.loc["32768+512x31", "heuristic_us"])


def _p3_vllm_best_over_grid(repo, data):
    grid = pd.read_csv(data / "dispatch_paged_cold.csv").set_index("lens")
    return float(grid.loc["32768+512x31", "heuristic_us"] / grid.loc["32768+512x31", "best_us"])


@lru_cache(maxsize=None)
def _p3_gsm8k(data: Path) -> pd.DataFrame:
    from kernelscope.analysis.accuracy import PERQUESTION
    return pd.read_csv(data / "serve_4090" / GSM8K_RUN / PERQUESTION)


def _p3_acc(policy, metric, key):
    def f(repo, data):
        from kernelscope.analysis.accuracy import policy_stats
        return float(policy_stats(_p3_gsm8k(data), policy, metric)[key])
    return f


def _p3_paired(policy, metric, key):
    def f(repo, data):
        from kernelscope.analysis.accuracy import paired
        return float(paired(_p3_gsm8k(data), policy, metric=metric)[key])
    return f


def _p3_gsm8k_events(policy, answered):
    """Divergence events of a policy in the 200 graded questions (answered) or in the 7 long filler requests."""
    def f(repo, data):
        import json as _json
        from kernelscope.serve.divergence import campaign_events
        events = campaign_events(data / "serve_4090" / GSM8K_RUN)
        gold = {int(k) for k in _json.loads((data / "serve_4090" / GSM8K_RUN / "gsm8k_ragged32k_n200_gold.json").read_text())["gold_by_rid"]}
        e = events[(events.policy == policy) & (events.repeat == 0)]
        return float(e.rid.isin(gold).sum() if answered else (~e.rid.isin(gold)).sum())
    return f



# ---- background figures for the talk (docs/presentation/background, 2026-10-25) ------------------------------------

BG = "docs/presentation/background/story.md"
BG_CELL = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"   # the headline mixed-length cell


def _bg_len(trace, key):
    def f(repo, data):
        d = json.loads((data / "background" / "prompt_length_hist.json").read_text())
        return float(next(t for t in d["traces"] if t["name"].startswith(trace))[key])
    return f


def _bg_len_all(key):
    def f(repo, data):
        d = json.loads((data / "background" / "prompt_length_hist.json").read_text())
        return float(sum(t[key] for t in d["traces"]))
    return f


def _bg_cta(variant, what):
    """CTA work of the headline cell on a 128-SM GPU (analysis of kernelscope.model.geometry.build_launch)."""
    def f(repo, data):
        from kernelscope.model.geometry import build_launch, parse_variant
        from kernelscope.workload import Workload
        keys = build_launch(Workload.from_key(BG_CELL), parse_variant(variant), 128).keys
        return {"ctas": float(len(keys)), "max": float(keys.max()), "max_over_sm_share": float(keys.max() / (keys.sum() / 128))}[what]
    return f


CHECKS = [
    Check("kernel.uniform_median","균일 배치(S1, dense, cold): 기본 휴리스틱의 최적 대비 손실 중앙값",
          0.72, 0.005, "%", _uniform_regret_pct("median_regret"), P0, "median 0.72 %, max 5.71 %"),
    Check("kernel.uniform_max", "균일 배치(S1, dense, cold): 기본 휴리스틱의 최적 대비 손실 최댓값",
          5.71, 0.005, "%", _uniform_regret_pct("max_regret"), P0, "median 0.72 %, max 5.71 %"),
    Check("kernel.ragged_worst_dense", "혼합 길이 최악 셀(32K×1 + 512×31): 휴리스틱/최적 커널 시간 비",
          12.53, 0.005, "x", _slowdown("ragged_s1_dense", "dense", WORST_RAGGED), P0, "12.53×"),
    Check("kernel.ragged_worst_paged", "같은 셀, paged KV 캐시: 휴리스틱/최적 커널 시간 비",
          3.84, 0.005, "x", _slowdown("ragged_s1_paged", "paged", WORST_RAGGED), P0, "3.84×"),
    Check("kernel.demo_cell", "발표 예시 셀(32K×2 + 1K×30): 휴리스틱/최적 커널 시간 비",
          6.85, 0.005, "x", _slowdown("ragged_s1_dense", "dense", DEMO_RAGGED), P0, "6.85× slower"),

    Check("serve.ragged_tpot_heuristic", "Qwen3-4B 혼합 길이: 기본 휴리스틱 TPOT",
          61.06, 0.005, "ms", _serve("ragged", "heuristic", "tpot_ms_mean"), "README.md", "61.06 → 33.80ms"),
    Check("serve.ragged_tpot_table", "Qwen3-4B 혼합 길이: 측정 테이블 정책 TPOT",
          33.80, 0.005, "ms", _serve("ragged", "table", "tpot_ms_mean"), "README.md", "61.06 → 33.80ms"),
    Check("serve.ragged_speedup", "Qwen3-4B 혼합 길이: 측정 테이블 정책의 TPOT 개선 배율",
          1.81, 0.005, "x", _serve("ragged", "table", "speedup_vs_heuristic"), "README.md", "(1.81배)"),
    Check("serve.ragged_attn_heuristic", "혼합 길이: step당 attention 시간(휴리스틱)",
          36.87, 0.005, "ms", _serve("ragged", "heuristic", "attn_ms_per_step"), "docs/demo.md", "36.87 → 9.24ms/step"),
    Check("serve.ragged_attn_table", "혼합 길이: step당 attention 시간(측정 테이블)",
          9.24, 0.005, "ms", _serve("ragged", "table", "attn_ms_per_step"), "docs/demo.md", "36.87 → 9.24ms/step"),
    Check("serve.ragged_decode_heuristic", "혼합 길이: step당 전체 decode 시간(휴리스틱)",
          51.36, 0.005, "ms", _serve("ragged", "heuristic", "decode_wall_ms_per_step"), "docs/demo.md",
          "51.36 → 24.08ms/step"),
    Check("serve.ragged_decode_table", "혼합 길이: step당 전체 decode 시간(측정 테이블)",
          24.08, 0.005, "ms", _serve("ragged", "table", "decode_wall_ms_per_step"), "docs/demo.md",
          "51.36 → 24.08ms/step"),
    Check("serve.ragged_tokens_identical", "혼합 길이: 측정 테이블 정책의 생성 토큰이 휴리스틱과 동일(1=동일)",
          1.0, 0.0, "", _serve("ragged", "table", "tokens_equivalent")),
    Check("serve.arrivals_tpot_heuristic", "요청 도착 시나리오: 기본 휴리스틱 TPOT",
          59.09, 0.005, "ms", _serve("arrivals", "heuristic", "tpot_ms_mean"), "README.md", "59.09 → 50.34ms"),
    Check("serve.arrivals_tpot_table", "요청 도착 시나리오: 측정 테이블 정책 TPOT",
          50.34, 0.005, "ms", _serve("arrivals", "table", "tpot_ms_mean"), "README.md", "59.09 → 50.34ms"),
    Check("serve.uniform_table_speedup", "균일 배치: 측정 테이블 정책의 TPOT 배율(개선 없음, 1.00±0.02)",
          1.00, 0.02, "x", _serve("uniform", "table", "speedup_vs_heuristic"), "docs/graduation.md", "거의 없었다"),
    Check("serve.uniform_fixed8_agreement", "균일 배치: 고정 split 8의 생성 토큰 일치율",
          94.43, 0.005, "%", _token_agreement_pct("uniform", "fixed8"), "docs/STATUS.md", "94.43%"),

    Check("followup.runs", "후속 실험: 최종 측정 실행 수(2모델 × 3조건 × 2시드 × 3정책 × 3반복)",
          108, 0, "runs", _followup_runs, "docs/STATUS.md", "108 final GPU runs"),
    Check("followup.qwen_ragged_min", "후속 실험 Qwen3-4B 혼합 길이: 모델 정책 개선 배율 최솟값",
          1.282, 0.0005, "x", _followup_speedup("Qwen3-4B", "min"), "docs/STATUS.md", "1.282–1.290×"),
    Check("followup.qwen_ragged_max", "후속 실험 Qwen3-4B 혼합 길이: 모델 정책 개선 배율 최댓값",
          1.290, 0.0005, "x", _followup_speedup("Qwen3-4B", "max"), "docs/STATUS.md", "1.282–1.290×"),
    Check("followup.llama_ragged_min", "후속 실험 Llama-8B 혼합 길이: 모델 정책 개선 배율 최솟값",
          1.184, 0.0005, "x", _followup_speedup("Llama-8B", "min"), "docs/STATUS.md", "1.184–1.188×"),
    Check("followup.llama_ragged_max", "후속 실험 Llama-8B 혼합 길이: 모델 정책 개선 배율 최댓값",
          1.188, 0.0005, "x", _followup_speedup("Llama-8B", "max"), "docs/STATUS.md", "1.184–1.188×"),
    Check("followup.output_valid", "후속 실험: 출력 검증을 통과한 비교 수(24개 중)",
          14, 0, "comparisons", _followup_valid, "docs/STATUS.md", "14/24"),

    Check("latency.python_cold", "모델 정책 첫 선택 시간, NumPy 구현(혼합 길이 step 0)",
          842.600, 0.0005, "ms", _cold_decision_ms("python"), "docs/STATUS.md", "842.600 → 14.293ms"),
    Check("latency.native_cold", "모델 정책 첫 선택 시간, C 구현(혼합 길이 step 0)",
          14.293, 0.0005, "ms", _cold_decision_ms("native"), "docs/STATUS.md", "842.600 → 14.293ms"),

    Check("model.v1_cold_ape", "대리 모델 V1(보간, cold): 실행 시간 예측 오차 중앙값",
          9.0, 0.05, "%", _ape_pct("V1", "cold"), VAL, "| V1 | cold | 594 | 9.0% |"),
    Check("model.v1_warm_ape", "대리 모델 V1(보간, warm): 실행 시간 예측 오차 중앙값",
          2.3, 0.05, "%", _ape_pct("V1", "warm"), VAL, "| V1 | warm | 594 | 2.3% |"),
    Check("model.v5_cold_ape", "대리 모델 V5(다른 head 구성): 실행 시간 예측 오차 중앙값",
          12.3, 0.05, "%", _ape_pct("V5", "cold"), VAL, "| V5 | cold | 882 | 12.3% |"),
    Check("model.v6_cold_ape", "대리 모델 V6(혼합 길이): 실행 시간 예측 오차 중앙값",
          11.1, 0.05, "%", _ape_pct("V6", "cold"), VAL, "| V6 | cold | 1971 | 11.1% |"),
    Check("model.v6_model_max_regret", "V6(혼합 길이): 대리 모델이 고른 커널의 최적 대비 손실 최댓값",
          24.5, 0.05, "%", _regret_pct("V6", "cold", "model_max_regret"), VAL, "0.0% / 24.5%"),
    Check("model.v6_heuristic_max_regret", "V6(혼합 길이): 기본 휴리스틱의 최적 대비 손실 최댓값",
          1152.7, 0.05, "%", _regret_pct("V6", "cold", "heuristic_max_regret"), VAL, "64.2% / 1152.7%"),

    Check("model.hybrid02_v1_max_regret", "혼합 정책(δ=0.2) V1 cold: 고른 커널의 최적 대비 손실 최댓값 (leave-one-out)",
          1.97, 0.005, "%", _hybrid_max_pct("V1", "cold"), HYB, "1.97%"),
    Check("model.hybrid02_v5_max_regret", "혼합 정책(δ=0.2) V5 cold: 손실 최댓값 (leave-one-out)",
          7.54, 0.005, "%", _hybrid_max_pct("V5", "cold"), HYB, "7.54%"),
    Check("model.hybrid02_v6_max_regret", "혼합 정책(δ=0.2) V6 cold: 손실 최댓값 (leave-one-out)",
          10.90, 0.005, "%", _hybrid_max_pct("V6", "cold"), HYB, "10.90%"),

    Check("sim.ragged_s16_speedup", "GPGPU-Sim(SM7_QV100) 축소 split-KV 커널, 혼합 길이 2048+128×15: S=1 대비 S=16 사이클 비",
          7.23, 0.005, "x", _sim_speedup("ragged", 16), SIM, "7.23배"),
    Check("sim.uniform_s8_speedup", "GPGPU-Sim 축소 split-KV 커널, 균일 길이 256×16: S=1 대비 S=8 사이클 비",
          2.25, 0.005, "x", _sim_speedup("uniform", 8), SIM, "2.25배"),
    Check("sim.ragged_vs_uniform_s1", "GPGPU-Sim S=1: 혼합 길이 배치 사이클 / 균일 길이 배치 사이클 (키 총량 비슷)",
          8.53, 0.005, "x", _sim_ratio_s1, SIM, "8.5배"),

    *[Check(f"final2.accelsim.{cell}.{what}", f"최종보고서 v2 표 10 {cell}: {title}", exp, tol, unit, fn(cell), FINAL2, row)
      for cell, row, values in [
          ("fa2_b1_l1k", "| FA2 B=1, L=1K | 63.584 | 67.5 | 1.061 | 61.0 | 4.1% | 밖(보간) |", (63.584, 1.061, 60.960, 4.127)),
          ("fa2_b1_l8k", "| FA2 B=1, L=8K | 494.336 | 515.9 | 1.044 | 460.1 | 6.9% | 안 |", (494.336, 1.0437, 460.061, 6.933)),
          ("fa2_b16_l1k", "| FA2 B=16, L=1K | 64.303 | 109.7 | 1.707 | 61.0 | 5.2% | 밖(보간) |", (64.303, 1.7065, 60.960, 5.199)),
          ("fd_b1_l1k", "| FlashDecoding B=1, L=1K | 10.016 | 15.5 | 1.548 | 10.8 | 8.0% | 밖(보간) |", (10.016, 1.5482, 10.821, 8.041)),
          ("fd_b1_l8k", "| FlashDecoding B=1, L=8K | 27.456 | 73.6 | 2.680 | 23.3 | 15.0% | 안 |", (27.456, 2.6803, 23.335, 15.008)),
          ("fd_b16_l1k", "| FlashDecoding B=16, L=1K | 38.2235 | 116.8 | 3.056 | 37.2 | 2.7% | 밖(보간) |", (38.2235, 3.0559, 37.207, 2.660))]
      for (what, title, unit, tol, fn), exp in zip([
          ("real_us", "실측 커널 시간(torch.profiler)", "us", 0.0005, lambda c: _accelsim_value(c, "kernel_time_us")),
          ("accelsim_ratio", "Accel-Sim 재생 / 실측", "x", 0.0005, lambda c: _accelsim_value(c, "sim_vs_kernel_time")),
          ("surrogate_us", "성능 시뮬레이터 warm 예측", "us", 0.0005, _surrogate_us),
          ("surrogate_err", "성능 시뮬레이터 예측 오차(|예측/실측 - 1|)", "%", 0.0005, lambda c: lambda repo, data: _accelsim_surrogate(repo, data)[c]["err_pct"])],
          values)],
    Check("final2.surrogate_err.min", "표 10 6개 조건: 성능 시뮬레이터 예측 오차 최솟값(warm)", 2.66, 0.005, "%", _surrogate_err("min"), FINAL2, "2.7~15.0%"),
    Check("final2.surrogate_err.max", "표 10 6개 조건: 성능 시뮬레이터 예측 오차 최댓값(warm)", 15.008, 0.0005, "%", _surrogate_err("max"), FINAL2, "2.7~15.0%"),
    Check("final2.surrogate_err.median", "표 10 6개 조건: 성능 시뮬레이터 예측 오차 중앙값(warm)", 6.07, 0.005, "%", _surrogate_err("median"), FINAL2, "중앙값 6.1%"),
    *[Check(f"final2.gpgpusim_wall.{w}_s{s}", f"GPGPU-Sim 축소 split-KV 커널 1회 실행 시간(sim.log gpgpu_simulation_time), {w} S={s}",
            exp, 0, "s", _sim_wall_s(w, s), FINAL2, f"| {w} | {s} | {exp} |")
      for w, s, exp in [("ragged", 1, 330), ("ragged", 2, 230), ("ragged", 4, 226), ("ragged", 8, 288), ("ragged", 16, 362),
                        ("uniform", 1, 219), ("uniform", 16, 351)]],

    *[Check(f"delta.{s}.d{int(round(d * 100)):03d}", f"혼합 정책 δ={d:g}, {s} cold: leave-one-out 최악 손실", exp, 0.005, "%",
            _delta_worst_pct(s, d), EVIDENCE, row)
      for d, row, values in [
          (0.0, "| 0.00 | 24.25% (1) | 66.29% (3) | 24.52% (2) |", (24.25, 66.29, 24.52)),
          (0.05, "| 0.05 | 24.25% (1) | 66.29% (3) | 7.53% (0) |", (24.25, 66.29, 7.53)),
          (0.1, "| 0.10 | 24.25% (1) | 66.29% (3) | 9.78% (0) |", (24.25, 66.29, 9.78)),
          (0.15, "| 0.15 | 1.97% (0) | 24.75% (1) | 10.90% (0) |", (1.97, 24.75, 10.90)),
          (0.16, "| 0.16 | 1.97% (0) | 7.54% (0) | 10.90% (0) |", (1.97, 7.54, 10.90)),
          (0.26, "| 0.26 | 1.97% (0) | 7.54% (0) | 10.90% (0) |", (1.97, 7.54, 10.90)),
          (0.3, "| 0.30 | 3.34% (0) | 7.54% (0) | 10.90% (0) |", (3.34, 7.54, 10.90)),
          (0.32, "| 0.32 | 3.34% (0) | 14.83% (0) | 10.90% (0) |", (3.34, 14.83, 10.90)),
          (0.33, "| 0.33 | 16.86% (1) | 14.83% (0) | 10.90% (0) |", (16.86, 14.83, 10.90)),
          (0.5, "| 0.50 | 16.86% (1) | 16.08% (1) | 10.90% (0) |", (16.86, 16.08, 10.90)),
          (1.0, "| 1.00 | 50.37% (7) | 66.07% (11) | 94.58% (1) |", (50.37, 66.07, 94.58))]
      for s, exp in zip(("V1", "V5", "V6"), values)],
    Check("delta.pass.low", "δ 격자(0.01 간격): cold 세 집합의 최악 손실이 모두 15% 이내인 가장 작은 δ", 0.16, 0, "", _delta_points("pass", "low"), EVIDENCE, "δ = 0.16~0.32"),
    Check("delta.pass.high", "δ 격자: cold 세 집합 모두 15% 이내인 가장 큰 δ", 0.32, 0, "", _delta_points("pass", "high"), EVIDENCE, "δ = 0.16~0.32"),
    Check("delta.pass.points", "δ 격자: 기준을 통과하는 점의 수(0.16~0.32가 끊김 없이 17점)", 17, 0, "", _delta_points("pass"), EVIDENCE, "17개 점"),
    Check("delta.flat.low", "δ 격자: 세 집합의 최악 손실이 δ=0.2와 같은 가장 작은 δ", 0.16, 0, "", _delta_points("flat", "low"), EVIDENCE, "0.16~0.26"),
    Check("delta.flat.high", "δ 격자: 세 집합의 최악 손실이 δ=0.2와 같은 가장 큰 δ", 0.26, 0, "", _delta_points("flat", "high"), EVIDENCE, "0.16~0.26"),
    Check("delta.flat.points", "δ 격자: δ=0.2와 최악 손실이 같은 점의 수(끊김 없이 11점)", 11, 0, "", _delta_points("flat"), EVIDENCE, "11개 점"),

    *[Check(f"threshold.{gpu.replace(' ', '_')}.hkv{h}", f"FA2 2.8.3 분할 없음 임계 배치: {gpu}, GPU당 KV 헤드 {h}개", exp, 0, "",
            _threshold(gpu, h), EVIDENCE, row)
      for gpu, row, values in [
          ("L4", "| L4 | 58 | 12 | 24 | 47 | 93 |", (12, 24, 47, 93)),
          ("RTX 4080 16GB", "| RTX 4080 16GB | 76 | 16 | 31 | 61 | 122 |", (16, 31, 61, 122)),
          ("RTX 3090", "| RTX 3090 | 82 | 17 | 33 | 66 | 132 |", (17, 33, 66, 132)),
          ("A100", "| A100 | 108 | 22 | 44 | 87 | 173 |", (22, 44, 87, 173)),
          ("H100 PCIe", "| H100 PCIe | 114 | 23 | 46 | 92 | 183 |", (23, 46, 92, 183)),
          ("RTX 4090", "| RTX 4090 | 128 | 26 | 52 | 103 | 205 |", (26, 52, 103, 205)),
          ("H100 SXM5", "| H100 SXM5 | 132 | 27 | 53 | 106 | 212 |", (27, 53, 106, 212)),
          ("L40", "| L40 | 142 | 29 | 57 | 114 | 228 |", (29, 57, 114, 228))]
      for h, exp in zip((8, 4, 2, 1), values)],
    *[Check(f"threshold.launches.{key}", f"RTX 4090 휴리스틱 실측 기록: {title}", exp, 0, "",
            lambda repo, data, key=key: float(_launch_agreement(data)[key]), EVIDENCE, text)
      for key, title, exp, text in [("rows", "decode 휴리스틱 실행 기록 수(헤드 구성 2종)", 785, "785건"),
                                    ("agree", "launch 수(분할 시 2회)가 규칙과 일치한 기록 수", 785, "785건 모두"),
                                    ("split", "규칙상 분할하는 기록 수", 426, "분할 426건"),
                                    ("no_split", "규칙상 분할하지 않는 기록 수", 359, "분할 없음 359건")]],

    *[Check(f"mooncake.len.{trace}.{stat}", f"Mooncake {trace} 트레이스(Kimi 토크나이저): {title}", exp, tol, "", _mooncake_lengths(trace, stat), EVIDENCE, row)
      for trace, row, values in [
          ("conversation", "| conversation | 12,031 | 6,909 | 27,367 | 85,400 | 126,195 | 6.9% | 2.1% | 350 |",
           (12031, 6909, 27367, 85400.4, 126195, 6.89, 2.11, 350)),
          ("arxiv", "| arxiv | 23,608 | 6,345 | 16,789 | 61,616 | 125,546 | 3.1% | 0.9% | 30 |",
           (23608, 6345, 16789.1, 61615.86, 125546, 3.06, 0.89, 30))]
      for (stat, title, tol), exp in zip([("rows", "요청 수", 0), ("p50", "입력 길이 중앙값", 0), ("p90", "입력 길이 p90", 0.5),
                                          ("p99", "입력 길이 p99", 0.5), ("max", "입력 길이 최댓값", 0),
                                          ("ge_32k_pct", "입력 32K 이상 비율(%)", 0.005), ("ge_64k_pct", "입력 64K 이상 비율(%)", 0.005),
                                          ("out_p50", "출력 길이 중앙값", 0)], values)],
    *[Check(f"mooncake.{run}.{key}", f"Mooncake 재생 {run}: {title}", exp, tol, "", _traffic(run, *path), EVIDENCE, row)
      for run, row, values in [
          ("mooncake_conv_x1", "| mooncake_conv_x1 | 10 GiB(72,817) | ×1 | 11,841 | 190 | 814,122 | 5.0 | 0.0% | 28.0% | 9.3% | 0.1% | 1.24 | 1.75 | 1.069 | 1.074 | 1.068 | 10,725.1 s |",
           (11841, 190, 814122, 4.9589, 0.0, 0.0932, 1.0686, 1.0742, 1.0684, 10725.09)),
          ("mooncake_conv_x0.5_kv436k", "| mooncake_conv_x0.5_kv436k | 436K | ×0.5 | 12,031 | 0 | 236,435 | 17.4 | 8.7% | 44.8% | 17.0% | 7.5% | 1.41 | 3.11 | 1.187 | 1.035 | 1.010 | 0.2 s |",
           (12031, 0, 236435, 17.3833, 0.0871, 0.1696, 1.1871, 1.0346, 1.0101, 0.18)),
          ("mooncake_conv_x1_kv436k", "| mooncake_conv_x1_kv436k | 436K | ×1 | 12,031 | 0 | 133,306 | 30.8 | 75.6% | 83.2% | 61.3% | 36.4% | 2.02 | 2.76 | 1.461 | 1.020 | 1.003 | 251.5 s |",
           (12031, 0, 133306, 30.8314, 0.7559, 0.6133, 1.4609, 1.0197, 1.0034, 251.5)),
          ("mooncake_arxiv_x1_kv436k", "| mooncake_arxiv_x1_kv436k | 436K | ×1 | 23,608 | 0 | 134,264 | 31.8 | 78.7% | 84.5% | 63.7% | 37.8% | 2.04 | 2.76 | 1.476 | 1.018 | 1.004 | 232.4 s |",
           (23608, 0, 134264, 31.8493, 0.7867, 0.6369, 1.4759, 1.0185, 1.0035, 232.36))]
      for (key, title, path, tol), exp in zip([
          ("requests", "재생한 요청 수", ("trace", "requests"), 0), ("too_large", "KV 예산을 넘어 건너뛴 요청 수", ("trace", "too_large"), 0),
          ("steps", "decode step 수", ("steps",), 0), ("batch_mean", "평균 배치(채점한 step)", ("losses", "batch_mean"), 0.00005),
          ("no_split", "휴리스틱이 분할하지 않는 step 비율", ("losses", "no_split_frac"), 0.00005),
          ("loss_ge_1.25", "예측 손실 1.25배 이상 step 비율", ("losses", "loss_ge_1.25_frac"), 0.00005),
          ("attention_ratio", "휴리스틱 / 최선 attention 시간 비(예측)", ("losses", "attention_time_ratio"), 0.00005),
          ("s8_ratio_best", "고정 분할 8 / 최선 attention 시간 비(예측)", ("fixed", "s8.ratio_best"), 0.00005),
          ("s16_ratio_best", "고정 분할 16 / 최선 attention 시간 비(예측)", ("fixed", "s16.ratio_best"), 0.00005),
          ("wait_mean_s", "도착부터 배치 진입까지 평균 대기(s)", ("load", "wait_mean_s"), 0.005)], values)],

    # ---- GPU optional experiments of 2026-10-28: (a) fixed-split ragged re-measurement, (b) trace e2e teacher forcing
    # ---- over 1,600 decode calls, (c) the split count vLLM 0.31 actually ran (FA2 launches in the 2026-10-25 traces)
    *[Check(f"p3rerun.ragged.{pol}.speedup", f"고정 분할 혼합 길이 재측정(2026-10-28): {pol}의 TPOT 배율(휴리스틱 / 정책, 3회 평균)", exp, 0.0005, "x",
            _p3_serve(FIXED_RERUN, "ragged", pol, "speedup_vs_heuristic"), EVIDENCE, row)
      for pol, exp, row in [("table", 1.8057, '| table | 1.806 | 1.739–1.864 | 34.66 / 35.08 / 33.91 | 9.45 |'), ("fixed8", 1.8234, '| 고정 8 | 1.823 | 1.792–1.840 | 34.21 / 34.05 / 34.38 | 9.32 |'), ("fixed16", 1.8139, '| 고정 16 | 1.814 | 1.782–1.867 | 33.72 / 34.23 / 35.23 | 9.40 |')]],
    Check("p3rerun.ragged.fixed8.ci_low", "재측정 고정 8의 반복 부트스트랩 구간 하한", 1.7918, 0.0005, "x",
          _p3_serve(FIXED_RERUN, "ragged", "fixed8", "speedup_ci95_low"), EVIDENCE, '| 고정 8 | 1.823 | 1.792–1.840 | 34.21 / 34.05 / 34.38 | 9.32 |'),
    Check("p3rerun.ragged.fixed8.ci_high", "재측정 고정 8의 반복 부트스트랩 구간 상한", 1.8396, 0.0005, "x",
          _p3_serve(FIXED_RERUN, "ragged", "fixed8", "speedup_ci95_high"), EVIDENCE, '| 고정 8 | 1.823 | 1.792–1.840 | 34.21 / 34.05 / 34.38 | 9.32 |'),
    *[Check(f"p3rerun.ragged.{pol}.repeat{i}_tpot", f"재측정 {pol} 반복 {i}의 TPOT(ms)", exp, 0.005, "ms",
            _p3_repeat(FIXED_RERUN, "ragged", pol, i, "tpot_ms"), EVIDENCE, row)
      for pol, values, row in [("heuristic", (62.937, 61.010, 63.204), '| heuristic | 1.000 | — | 62.94 / 61.01 / 63.20 | 37.68 |'), ("table", (34.657, 35.083, 33.905), '| table | 1.806 | 1.739–1.864 | 34.66 / 35.08 / 33.91 | 9.45 |'),
                               ("fixed8", (34.212, 34.049, 34.377), '| 고정 8 | 1.823 | 1.792–1.840 | 34.21 / 34.05 / 34.38 | 9.32 |'), ("fixed16", (33.717, 34.234, 35.227), '| 고정 16 | 1.814 | 1.782–1.867 | 33.72 / 34.23 / 35.23 | 9.40 |')]
      for i, exp in enumerate(values)],
    *[Check(f"p3rerun.ragged.{pol}.attn", f"재측정 {pol}의 attention ms/step", exp, 0.005, "ms",
            _p3_serve(FIXED_RERUN, "ragged", pol, "attn_ms_per_step"), EVIDENCE, row)
      for pol, exp, row in [("heuristic", 37.680, '| heuristic | 1.000 | — | 62.94 / 61.01 / 63.20 | 37.68 |'), ("table", 9.4503, '| table | 1.806 | 1.739–1.864 | 34.66 / 35.08 / 33.91 | 9.45 |'), ("fixed8", 9.3182, '| 고정 8 | 1.823 | 1.792–1.840 | 34.21 / 34.05 / 34.38 | 9.32 |'), ("fixed16", 9.4020, '| 고정 16 | 1.814 | 1.782–1.867 | 33.72 / 34.23 / 35.23 | 9.40 |')]],
    Check("p3rerun.ragged.tpot_spread", "재측정: 휴리스틱을 뺀 세 정책의 반복 9개 TPOT 최대 / 최소", 1.0448, 0.0005, "x", _rerun_tpot_spread,
          EVIDENCE, "최대/최소가 1.045배 안이다"),
    Check("p3rerun.ragged.mismatches", "재측정: 세 정책·세 반복의 휴리스틱 대비 불일치 토큰 수 합", 0, 0, "", _rerun_mismatches, EVIDENCE, "(불일치 0)"),
    *[Check(f"p3tracetf.{pol}.{col}", f"트레이스 e2e {pol}의 분기 지점, 교사 강제 1,600 호출로 분류: {title}", exp, 0, "", _trace_tf(pol, col), EVIDENCE, row)
      for pol, row, values in [("table", "| table | 10 | 10 | 0 | 0 | 0 | 0 | 15 |", (10, 10, 0, 0, 0, 0)),
                               ("hybrid", "| hybrid(δ=0.2) | 10 | 10 | 0 | 0 | 0 | 0 | 15 |", (10, 10, 0, 0, 0, 0)),
                               ("model", "| model | 8 | 7 | 1 | 0 | 0 | 0 | 17 |", (8, 7, 1, 0, 0, 0))]
      for (col, title), exp in zip([("distinct_positions", "분기 지점 수"), ("tie_1ulp", "동점 1 ulp"), ("tie_2ulp", "동점 2 ulp"),
                                    ("clear", "clear"), ("unclassified", "미분류"), ("not_reproduced", "재현 안 됨")], values)],
    *[Check(f"p3tracetf.{pol}.flips", f"트레이스 e2e 교사 강제 1,600 호출: {pol}의 argmax가 바뀐 행 수", exp, 0, "", _trace_tf_flips(pol), EVIDENCE, row)
      for pol, exp, row in [("table", 15, "| table | 10 | 10 | 0 | 0 | 0 | 0 | 15 |"), ("hybrid", 15, "| hybrid(δ=0.2) | 10 | 10 | 0 | 0 | 0 | 0 | 15 |"),
                            ("model", 17, "| model | 8 | 7 | 1 | 0 | 0 | 0 | 17 |")]],
    *[Check(f"p3tracetf.{pol}.rows", f"트레이스 e2e 교사 강제 1,600 호출: {pol}의 기록 행 수(전체 파일)", 45409, 0, "", _trace_tf_summary("policies", pol, "rows"),
            EVIDENCE, "정책마다 45,409행을 기록했다") for pol in ("table", "hybrid", "model")],
    Check("p3tracetf.calls", "트레이스 e2e 교사 강제: 기록한 decode 호출 수", 1600, 0, "", _trace_tf_summary("decode_calls_recorded"), EVIDENCE, "1,600 호출로 늘렸다"),
    Check("p3tracetf.first400_identical", "처음 400 호출의 10월 24일 행(세 정책 26,052행) 가운데 모든 열이 같은 행 수", 26052, 0, "",
          _trace_tf_summary("first_400_calls", "identical_rows"), EVIDENCE, "8,684행은 10월 24일 기록과 값이 모두 같다"),
    Check("p3tracetf.table.flips_first400", "처음 400 호출 안에서 table의 argmax가 바뀐 행(10월 24일과 같은 1행)", 1, 0, "", _trace_tf_flips("table", 400),
          EVIDENCE, "8,684행은 10월 24일 기록과 값이 모두 같다"),
    Check("p3tracetf.table.diff_gt4", "argmax는 같지만 로짓 최대 차이가 4를 넘는 table 행 수", 247, 0, "", _trace_tf_summary("policies", "table", "rows_diff_gt_4"),
          EVIDENCE, "table 247행(최대 18.6"),
    Check("p3tracetf.table.max_diff", "table 교사 강제 행의 로짓 최대 차이", 18.64, 0.005, "", _trace_tf_summary("policies", "table", "max_abs_logit_diff"),
          EVIDENCE, "table 247행(최대 18.6"),
    *[Check(f"vllmsplit.{run}.{what}", f"vLLM 0.31 프로파일 {run}: {title}", exp, tol, unit, _fa2_launch(run, what), EVIDENCE, row)
      for run, row, values in [
          ("flash_attn_eager_ragged", "| FA2 eager, 혼합 | 792 | 0 | 0 | 0 | 1,875.8 |", (792, 0, 0, 0, 1875.8)),
          ("flash_attn_eager_uniform", "| FA2 eager, 균일 | 792 | 0 | 0 | 0 | 81.4 |", (792, 0, 0, 0, 81.4)),
          ("flash_attn_cudagraph_ragged", "| FA2 CUDA graph, 혼합 | 828 | 828 | 0 | 0 | 1,873.6 |", (828, 828, 0, 0, 1873.6)),
          ("flash_attn_cudagraph_uniform", "| FA2 CUDA graph, 균일 | 828 | 828 | 0 | 0 | 79.1 |", (828, 828, 0, 0, 79.1))]
      for (what, title, tol, unit), exp in zip([("decode", "배치 32 decode의 FA2 실행 수(grid (1, 32, 8))", 0, ""),
                                                 ("graph", "그중 CUDA graph 안에서 실행된 수", 0, ""),
                                                 ("split", "Split=true인 FA2 실행 수(모든 grid)", 0, ""),
                                                 ("combine", "flash_fwd_splitkv_combine_kernel 실행 수", 0, ""),
                                                 ("dur", "decode FA2 커널 시간 중앙값", 0.05, "us")], values)],

    Check("consistency.amdahl_ragged", "혼합 길이: attention 시간 변화만으로 예측한 step 시간과 실측의 차이(±3%)",
          0.0, 3.0, "%", _amdahl_error_pct("ragged")),
    Check("consistency.amdahl_arrivals", "요청 도착: attention 시간 변화만으로 예측한 step 시간과 실측의 차이(±3%)",
          0.0, 3.0, "%", _amdahl_error_pct("arrivals")),
    Check("consistency.amdahl_uniform", "균일 배치: attention 시간 변화만으로 예측한 step 시간과 실측의 차이(±3%)",
          0.0, 3.0, "%", _amdahl_error_pct("uniform")),

    Check("diagnose.ragged_attention_share_heuristic", "혼합 길이 연산 분해(D1): 휴리스틱 step의 attention 비중",
          70.9, 0.5, "%", _diag_op("ragged", "heuristic", "attention", "share", 100), OPB, "70.9%"),
    Check("diagnose.ragged_attention_share_table", "혼합 길이 연산 분해(D1): 측정 테이블 정책 step의 attention 비중",
          37.5, 0.5, "%", _diag_op("ragged", "table", "attention", "share", 100), OPB, "37.5%"),
    Check("diagnose.ragged_unattributed_heuristic", "혼합 길이 연산 분해(D1): 미귀속 시간 비율(상한 10%, 5±5)",
          5.0, 5.0, "%", _diag_value("ragged", "heuristic", "unattributed_pct")),
    Check("diagnose.ragged_mlp_pct_dram_heuristic", "혼합 길이 연산 분해(D1): mlp 클래스의 DRAM 상한 대비 달성률(하한 추정)",
          86.1, 1.0, "%", _diag_op("ragged", "heuristic", "mlp", "pct_dram"), OPB, "86.1%"),
    Check("diagnose.ragged_amdahl_bound_heuristic", "혼합 길이 연산 분해(D1): attention만 최적으로 바꿀 때 step 상한 배율",
          2.10, 0.01, "x", _diag_value("ragged", "heuristic", "attention", "amdahl_bound"), OPB, "2.10배"),
    Check("diagnose.uniform_attention_share_heuristic", "균일 길이 연산 분해(D2): 휴리스틱 step의 attention 비중",
          19.2, 0.5, "%", _diag_op("uniform", "heuristic", "attention", "share", 100), OPB, "19.2%"),
    Check("diagnose.timer_overhead_max", "연산 분해 타이머 오버헤드 최댓값(세 실행, 상한 5%, 0±5)",
          0.0, 5.0, "%", _diag_overhead_max),

    Check("hybrid.ragged_tpot_heuristic", "혼합 정책 실생성 검증(2026-10-02), 혼합 길이: 기본 휴리스틱 TPOT",
          61.11, 0.005, "ms", _hybrid_serve("ragged", "heuristic", "tpot_ms_mean"), HYBV, "61.11 → 34.09ms"),
    Check("hybrid.ragged_tpot_hybrid", "혼합 길이: 혼합 정책(δ=0.2) TPOT",
          34.09, 0.005, "ms", _hybrid_serve("ragged", "hybrid", "tpot_ms_mean"), HYBV, "61.11 → 34.09ms"),
    Check("hybrid.ragged_tpot_table", "혼합 길이: 측정 테이블 정책 TPOT(같은 실행)",
          33.82, 0.005, "ms", _hybrid_serve("ragged", "table", "tpot_ms_mean"), HYBV, "33.82ms"),
    Check("hybrid.ragged_speedup_hybrid", "혼합 길이: 혼합 정책의 TPOT 개선 배율",
          1.793, 0.0005, "x", _hybrid_serve("ragged", "hybrid", "speedup_vs_heuristic"), HYBV, "1.793배"),
    Check("hybrid.ragged_speedup_table", "혼합 길이: 측정 테이블 정책의 TPOT 개선 배율(같은 실행)",
          1.807, 0.0005, "x", _hybrid_serve("ragged", "table", "speedup_vs_heuristic"), HYBV, "1.807배"),
    Check("hybrid.uniform_speedup_hybrid", "균일 길이: 혼합 정책의 TPOT 배율(선택 비용만큼 손해)",
          0.993, 0.0005, "x", _hybrid_serve("uniform", "hybrid", "speedup_vs_heuristic"), HYBV, "0.993배"),
    Check("hybrid.arrivals_speedup_hybrid", "요청 도착: 혼합 정책의 TPOT 개선 배율",
          1.146, 0.0005, "x", _hybrid_serve("arrivals", "hybrid", "speedup_vs_heuristic"), HYBV, "1.146배"),
    Check("hybrid.arrivals_speedup_table", "요청 도착: 측정 테이블 정책의 TPOT 개선 배율(같은 실행)",
          1.164, 0.0005, "x", _hybrid_serve("arrivals", "table", "speedup_vs_heuristic"), HYBV, "1.164배"),
    Check("hybrid.requested_steps_differing_from_table", "요청한 3조건 × 3회: 혼합 정책과 테이블 정책의 분할 수가 다른 decode step 수",
          0, 0, "steps", _steps_differing(REQUESTED), HYBV, "0/567"),
    Check("hybrid.ragged_policy_us_hybrid", "혼합 길이: 혼합 정책의 step당 선택 비용(실행마다 정책 캐시를 비움)",
          245.8, 0.05, "µs", _hybrid_serve("ragged", "hybrid", "policy_us_per_step"), HYBV, "245.8"),
    Check("hybrid.arrivals_policy_us_hybrid", "요청 도착: 혼합 정책의 step당 선택 비용",
          751.8, 0.05, "µs", _hybrid_serve("arrivals", "hybrid", "policy_us_per_step"), HYBV, "751.8"),
    Check("hybrid.heldout_steps_differing_from_table", "held-out 자연어 요청 도착: 55 step 중 혼합 정책이 테이블과 다르게 고른 step 수(3회 동일)",
          12, 0, "steps", _steps_differing(("heldout_arrivals",), per_repeat=True), HYBV, "12/55"),
    Check("hybrid.heldout_attn_heuristic", "held-out 요청 도착: step당 attention 시간(휴리스틱)",
          4.85, 0.005, "ms", _hybrid_serve("heldout_arrivals", "heuristic", "attn_ms_per_step"), HYBV, "4.85"),
    Check("hybrid.heldout_attn_table", "held-out 요청 도착: step당 attention 시간(측정 테이블)",
          3.31, 0.005, "ms", _hybrid_serve("heldout_arrivals", "table", "attn_ms_per_step"), HYBV, "3.31"),
    Check("hybrid.heldout_attn_hybrid", "held-out 요청 도착: step당 attention 시간(혼합 정책)",
          3.00, 0.005, "ms", _hybrid_serve("heldout_arrivals", "hybrid", "attn_ms_per_step"), HYBV, "3.00"),
    Check("hybrid.heldout_attn_model", "held-out 요청 도착: step당 attention 시간(모델 정책)",
          2.93, 0.005, "ms", _hybrid_serve("heldout_arrivals", "model", "attn_ms_per_step"), HYBV, "2.93"),
    Check("hybrid.heldout_speedup_table", "held-out 요청 도착: 측정 테이블 정책의 TPOT 배율",
          1.085, 0.0005, "x", _hybrid_serve("heldout_arrivals", "table", "speedup_vs_heuristic"), HYBV, "1.085배"),
    Check("hybrid.heldout_speedup_hybrid", "held-out 요청 도착: 혼합 정책의 TPOT 배율",
          1.020, 0.0005, "x", _hybrid_serve("heldout_arrivals", "hybrid", "speedup_vs_heuristic"), HYBV, "1.020배"),
    Check("hybrid.heldout_speedup_model", "held-out 요청 도착: 모델 정책의 TPOT 배율",
          1.028, 0.0005, "x", _hybrid_serve("heldout_arrivals", "model", "speedup_vs_heuristic"), HYBV, "1.028배"),
    Check("hybrid.heldout_policy_us_hybrid", "held-out 요청 도착: 혼합 정책의 step당 선택 비용",
          1561.1, 0.05, "µs", _hybrid_serve("heldout_arrivals", "hybrid", "policy_us_per_step"), HYBV, "1561.1"),

    Check("hybrid.requested_cache_misses", "요청한 3조건: 실행당 정책 캐시 미스(첫 결정) 횟수 합(혼합 1 + 균일 1 + 요청 도착 5, 3회 동일)",
          7, 0, "misses", _cache_misses(REQUESTED), HYBV, "1·1·5회"),
    Check("hybrid.heldout_cache_misses", "held-out 요청 도착: 실행당 정책 캐시 미스 횟수(3회 동일)",
          13, 0, "misses", _cache_misses(("heldout_arrivals",)), HYBV, "13회"),
    Check("hybrid.reproducibility_max_change_pct", "2026-09-22 캠페인 대비 휴리스틱·테이블 TPOT의 최대 변화율(3조건)",
          1.25, 0.005, "%", _reproducibility_pct, HYBV, "1.25%"),

    Check("hybrid.keep_heldout_speedup_table", "held-out 요청 도착, 정책 캐시 유지 규약(2026-10-06): 측정 테이블 정책의 TPOT 배율",
          1.086, 0.0005, "x", _keep_serve("heldout_arrivals", "table", "speedup_vs_heuristic"), HYBV, "1.086배"),
    Check("hybrid.keep_heldout_speedup_hybrid", "held-out 요청 도착, 캐시 유지: 혼합 정책의 TPOT 배율",
          1.090, 0.0005, "x", _keep_serve("heldout_arrivals", "hybrid", "speedup_vs_heuristic"), HYBV, "1.090배"),
    Check("hybrid.keep_heldout_speedup_model", "held-out 요청 도착, 캐시 유지: 모델 정책의 TPOT 배율",
          1.091, 0.0005, "x", _keep_serve("heldout_arrivals", "model", "speedup_vs_heuristic"), HYBV, "1.091배"),
    Check("hybrid.keep_heldout_tpot_hybrid", "held-out 요청 도착, 캐시 유지: 혼합 정책 TPOT",
          31.62, 0.005, "ms", _keep_serve("heldout_arrivals", "hybrid", "tpot_ms_mean"), HYBV, "31.62ms"),
    Check("hybrid.keep_heldout_policy_us_hybrid", "held-out 요청 도착, 캐시 유지: 혼합 정책의 step당 선택 비용",
          14.95, 0.005, "µs", _keep_serve("heldout_arrivals", "hybrid", "policy_us_per_step"), HYBV, "14.95"),
    Check("hybrid.keep_heldout_cache_misses", "held-out 요청 도착, 캐시 유지: 세 반복의 혼합 정책 캐시 미스 합(warm-up이 전부 흡수)",
          0, 0, "misses", _keep_cache_misses("heldout_arrivals", "hybrid"), HYBV, "캐시 미스 0회"),

    Check("flashinfer.ragged_cells", "FlashInfer 비교, 혼합 길이 격자(cold): 두 FlashInfer 변형과 FA2 변형이 모두 측정된 셀 수",
          162, 0, "cells", _fi_cells(True), HYBV, "162셀"),
    Check("flashinfer.uniform_cells", "FlashInfer 비교, 균일 길이 격자(cold): 측정 셀 수",
          49, 0, "cells", _fi_cells(False), HYBV, "49셀"),
    Check("flashinfer.ragged_cudacore_over_best_fa2_median", "혼합 길이: FlashInfer CUDA-core 변형 시간 / 최선 FA2 분할 시간의 중앙값",
          0.977, 0.0005, "x", _fi_ratio(True, "cudacore", "best_fa2", "median"), HYBV, "0.977배"),
    Check("flashinfer.ragged_cudacore_over_best_fa2_max", "혼합 길이: FlashInfer CUDA-core / 최선 FA2의 최댓값",
          1.003, 0.0005, "x", _fi_ratio(True, "cudacore", "best_fa2", "max"), HYBV, "1.003배"),
    Check("flashinfer.ragged_tensorcore_over_best_fa2_median", "혼합 길이: FlashInfer tensor-core 변형 / 최선 FA2의 중앙값",
          0.998, 0.0005, "x", _fi_ratio(True, "tensorcore", "best_fa2", "median"), HYBV, "0.998배"),
    Check("flashinfer.ragged_tensorcore_over_best_fa2_max", "혼합 길이: FlashInfer tensor-core 변형 / 최선 FA2의 최댓값",
          2.401, 0.0005, "x", _fi_ratio(True, "tensorcore", "best_fa2", "max"), HYBV, "2.401배"),
    Check("flashinfer.ragged_heuristic_over_best_flashinfer_max", "혼합 길이: 라이브러리 휴리스틱 / 더 빠른 FlashInfer 변형의 최댓값",
          4.11, 0.005, "x", _fi_ratio(True, "heuristic", "best_flashinfer", "max"), HYBV, "4.11배"),
    Check("flashinfer.uniform_best_over_best_fa2_median", "균일 길이: 더 빠른 FlashInfer 변형 / 최선 FA2의 중앙값",
          0.998, 0.0005, "x", _fi_ratio(False, "best_flashinfer", "best_fa2", "median"), HYBV, "0.998배"),
    Check("flashinfer.uniform_cudacore_over_best_fa2_max", "균일 길이: FlashInfer CUDA-core / 최선 FA2의 최댓값",
          1.181, 0.0005, "x", _fi_ratio(False, "cudacore", "best_fa2", "max"), HYBV, "1.181배"),
    Check("flashinfer.headline_cudacore_us", "최악 셀(32K×1 + 512×31, 페이지 KV): FlashInfer CUDA-core 시간",
          269.9, 0.05, "µs", _fi_headline("cudacore"), HYBV, "269.9"),
    Check("flashinfer.headline_tensorcore_us", "최악 셀: FlashInfer tensor-core 시간",
          670.9, 0.05, "µs", _fi_headline("tensorcore"), HYBV, "670.9"),

    Check("divergence.keep_heldout_tie_1ulp", "held-out 요청 도착, 캐시 유지 규약: 세 정책 사건 합(전부 tie_1ulp, 조사 항목 0)",
          6, 0, "events", _keep_divergence, HYBV, "캐시를 유지해도 사건 6건 모두 `tie_1ulp`"),

    Check("divergence.requested_events", "요청한 3조건: 테이블·혼합 정책의 분기 사건 수(서로 다른 위치)",
          0, 0, "events", _divergence("distinct_positions", REQUESTED), HYBV, "분기 사건 0건"),
    Check("divergence.requested_teacher_flips", "요청한 3조건: 교사 강제 진단에서 argmax가 뒤바뀐 위치 수",
          0, 0, "positions", _teacher_flips(REQUESTED), HYBV, "뒤바뀐 위치 0개"),
    Check("divergence.heldout_events_per_policy", "held-out 요청 도착: 정책별 분기 사건 수(table·hybrid·model 모두 같음)",
          2, 0, "events", _divergence("distinct_positions", ("heldout_arrivals",), agg="same"), HYBV, "정책별 2건"),
    Check("divergence.heldout_tie_1ulp", "held-out 요청 도착: 세 정책 사건 6건 중 tie_1ulp",
          6, 0, "events", _divergence("tie_1ulp", ("heldout_arrivals",)), HYBV, "6건이 모두 `tie_1ulp`"),
    Check("divergence.heldout_needs_investigation", "held-out 요청 도착: clear·미분류·미재현·토큰 누락·사건 밖 불일치의 합",
          0, 0, "events", _divergence(INVESTIGATE, ("heldout_arrivals",))),
    Check("divergence.control_events", "대조군(균일, fixed:8): 분기 사건 수",
          6, 0, "events", _divergence("distinct_positions", ("control_uniform_fixed8",)), HYBV, "분기 사건 6건"),
    Check("divergence.control_tie_1ulp", "대조군: tie_1ulp 사건 수",
          5, 0, "events", _divergence("tie_1ulp", ("control_uniform_fixed8",)), HYBV, "5건"),
    Check("divergence.control_clear", "대조군: clear 사건 수",
          1, 0, "events", _divergence("clear", ("control_uniform_fixed8",)), HYBV, "`clear` 1건"),
    Check("divergence.control_clear_margin_ulps", "대조군 clear 사건: 기준 1위 로짓과 후보가 고른 토큰의 로짓 차이(bf16 간격 단위)",
          70, 0, "ulp", _clear_margin_ulps("control_uniform_fixed8"), HYBV, "70"),
    Check("divergence.ragged_max_logit_diff", "혼합 길이 교사 강제 진단: 정책 간 로짓 최대 차이(32K 요청의 뒤 단계)",
          5.41, 0.005, "", _teacher_max_diff("ragged"), HYBV, "5.41"),
    Check("divergence.ragged_rows_identical_pct", "혼합 길이 교사 강제 진단: 로짓이 완전히 같은 위치의 비율",
          96.9, 0.05, "%", _teacher_identical_pct("ragged"), HYBV, "96.9%"),
    Check("divergence.control_max_logit_diff", "대조군 교사 강제 진단: 정책 간 로짓 최대 차이(clear 사건 위치)",
          22.625, 0.0005, "", _teacher_max_diff("control_uniform_fixed8"), HYBV, "22.625"),
    Check("divergence.control_probe_kernel_err", "대조군 clear 사건 step, 같은 KV 상태: 분할 1·8 attention 출력과 float32 참조의 최대 차이(36층)",
          0.134, 0.0005, "", _probe_kernel_err, HYBV, "0.134"),
    Check("divergence.control_probe_split_diff", "같은 KV 상태: 분할 1 출력과 분할 8 출력의 최대 차이(36층)",
          0.0625, 0, "", _probe_split_diff, HYBV, "0.0625"),
    Check("divergence.control_probe_single_step_logit_diff_rid8", "같은 KV 상태에서 한 step만 분할 8로 계산한 요청 8의 로짓 최대 차이",
          1.08, 0.005, "", _probe_single_step_logit_diff_rid8, HYBV, "1.08"),
    Check("divergence.control_probe_single_step_argmax_changes", "같은 KV 상태에서 한 step만 분할 8로 계산할 때 argmax가 바뀐 요청 수(32개 중)",
          0, 0, "requests", _probe_single_step_argmax_changes, HYBV, "바뀐 요청 0개"),
    Check("vllm.ragged_step_ms_flash_attn_eager", "vLLM 0.31 재현(eager, 기본 블록 16): FLASH_ATTN 백엔드, 혼합 길이 32K×1+512×31 배치의 step당 decode 시간",
          80.52, 0.005, "ms", _vllm_step_ms("repro2", "FLASH_ATTN", "eager", "ragged"), HYBV, "80.52"),
    Check("vllm.ragged_step_ms_flashinfer_eager", "vLLM 재현(eager): FLASHINFER 백엔드, 혼합 길이 step당 decode 시간",
          80.27, 0.005, "ms", _vllm_step_ms("repro2", "FLASHINFER", "eager", "ragged"), HYBV, "80.27"),
    Check("vllm.uniform_step_ms_flash_attn_eager", "vLLM 재현(eager): FLASH_ATTN 백엔드, 균일 512×32 배치의 step당 decode 시간",
          14.74, 0.005, "ms", _vllm_step_ms("repro2", "FLASH_ATTN", "eager", "uniform"), HYBV, "14.74"),
    Check("vllm.ragged_over_uniform_flash_attn_eager", "vLLM 재현(eager, FLASH_ATTN): 혼합 길이 / 균일 길이 step 시간 비",
          5.46, 0.005, "x", _vllm_ratio("repro2", ("FLASH_ATTN", "eager", "ragged"), ("FLASH_ATTN", "eager", "uniform")), HYBV, "5.46배"),
    Check("vllm.ragged_flashinfer_over_flash_attn_eager", "vLLM 재현(eager): 혼합 길이에서 FLASHINFER / FLASH_ATTN step 시간 비(백엔드 무관)",
          0.997, 0.0005, "x", _vllm_ratio("repro2", ("FLASHINFER", "eager", "ragged"), ("FLASH_ATTN", "eager", "ragged")), HYBV, "0.997배"),
    Check("vllm.ragged_step_ms_flash_attn_cudagraph", "vLLM 재현(CUDA Graph 모드, 분할 수 고정 경로): FLASH_ATTN 혼합 길이 step 시간",
          78.94, 0.005, "ms", _vllm_step_ms("repro3", "FLASH_ATTN", "cudagraph", "ragged"), HYBV, "78.94"),
    Check("vllm.ragged_step_ms_flash_attn_eager_block256", "vLLM 재현(eager, KV 블록 256): FLASH_ATTN 혼합 길이 step 시간",
          81.27, 0.005, "ms", _vllm_step_ms("repro4", "FLASH_ATTN", "eager", "ragged"), HYBV, "81.27"),

    Check("defaults.32k.ragged_cells", "정적 기본값 비교, 32K 격자 혼합 길이(cold): FA2 분할 변형과 FlashInfer 두 변형이 모두 측정된 셀 수",
          162, 0, "", _defaults_summary("32k", "ragged", "fa2_heuristic", "n"), SCOPE, "162셀"),
    Check("defaults.32k.fa2_heuristic_median", "32K 격자 혼합 길이: FA2 휴리스틱 / 전체 최선(FlashInfer 포함) 손실 중앙값",
          1.457, 0.0005, "x", _defaults_summary("32k", "ragged", "fa2_heuristic", "loss_median"), SCOPE, "1.457배"),
    Check("defaults.32k.fa2_heuristic_over_1.25", "32K 격자 혼합 길이: FA2 휴리스틱 손실이 1.25배를 넘는 셀 수",
          118, 0, "", _defaults_summary("32k", "ragged", "fa2_heuristic", "over_1.25"), SCOPE, "118셀"),
    Check("defaults.32k.flashinfer_tc_median", "32K 격자 혼합 길이: FlashInfer tensor-core 기본값 / 전체 최선 손실 중앙값",
          1.017, 0.0005, "x", _defaults_summary("32k", "ragged", "flashinfer_tc", "loss_median"), SCOPE, "1.017배"),
    Check("defaults.32k.flashinfer_tc_p90", "32K 격자 혼합 길이: FlashInfer tensor-core 손실 90퍼센타일",
          1.489, 0.0005, "x", _defaults_summary("32k", "ragged", "flashinfer_tc", "loss_p90"), SCOPE, "1.489배"),
    Check("defaults.32k.flashinfer_tc_max", "32K 격자 혼합 길이: FlashInfer tensor-core 손실 최댓값",
          2.485, 0.0005, "x", _defaults_summary("32k", "ragged", "flashinfer_tc", "loss_max"), SCOPE, "2.485배"),
    Check("defaults.32k.flashinfer_tc_over_1.25", "32K 격자 혼합 길이: FlashInfer tensor-core 손실이 1.25배를 넘는 셀 수",
          36, 0, "", _defaults_summary("32k", "ragged", "flashinfer_tc", "over_1.25"), SCOPE, "36셀"),
    Check("defaults.32k.flashinfer_cc_max", "32K 격자 혼합 길이: FlashInfer CUDA-core 변형 / 전체 최선 손실 최댓값",
          1.014, 0.0005, "x", _defaults_summary("32k", "ragged", "flashinfer_cc", "loss_max"), SCOPE, "1.014배"),
    Check("defaults.32k.fa2_best_median", "32K 격자 혼합 길이: FA2 분할만 고르는 선택기 / 전체 최선 손실 중앙값",
          1.025, 0.0005, "x", _defaults_summary("32k", "ragged", "fa2_best", "loss_median"), SCOPE, "1.025배"),
    Check("defaults.32k.fa2_best_max", "32K 격자 혼합 길이: FA2 분할만 고르는 선택기 손실 최댓값",
          1.063, 0.0005, "x", _defaults_summary("32k", "ragged", "fa2_best", "loss_max"), SCOPE, "1.063배"),
    Check("defaults.32k.uniform_fa2_heuristic_max", "32K 격자 균일 길이: FA2 휴리스틱 손실 최댓값",
          1.171, 0.0005, "x", _defaults_summary("32k", "uniform", "fa2_heuristic", "loss_max"), SCOPE, "1.171배"),
    Check("defaults.32k.uniform_flashinfer_tc_max", "32K 격자 균일 길이: FlashInfer tensor-core 손실 최댓값",
          1.053, 0.0005, "x", _defaults_summary("32k", "uniform", "flashinfer_tc", "loss_max"), SCOPE, "1.053배"),
    Check("defaults.64k.ragged_cells", "64K 격자 혼합 길이(cold): 측정 셀 수",
          18, 0, "", _defaults_summary("64k", "ragged", "fa2_heuristic", "n"), SCOPE, "18셀"),
    Check("defaults.64k.fa2_heuristic_median", "64K 격자 혼합 길이: FA2 휴리스틱 / 전체 최선 손실 중앙값",
          2.861, 0.0005, "x", _defaults_summary("64k", "ragged", "fa2_heuristic", "loss_median"), SCOPE, "2.861배"),
    Check("defaults.64k.fa2_heuristic_max", "64K 격자 혼합 길이: FA2 휴리스틱 손실 최댓값",
          5.265, 0.0005, "x", _defaults_summary("64k", "ragged", "fa2_heuristic", "loss_max"), SCOPE, "5.265배"),
    Check("defaults.64k.fa2_heuristic_over_2", "64K 격자 혼합 길이: FA2 휴리스틱 손실이 2배를 넘는 셀 수(18셀 전부)",
          18, 0, "", _defaults_summary("64k", "ragged", "fa2_heuristic", "over_2"), SCOPE, "18셀 전부"),
    Check("defaults.64k.flashinfer_tc_median", "64K 격자 혼합 길이: FlashInfer tensor-core 손실 중앙값",
          1.634, 0.0005, "x", _defaults_summary("64k", "ragged", "flashinfer_tc", "loss_median"), SCOPE, "1.634배"),
    Check("defaults.64k.flashinfer_tc_max", "64K 격자 혼합 길이: FlashInfer tensor-core 손실 최댓값",
          2.989, 0.0005, "x", _defaults_summary("64k", "ragged", "flashinfer_tc", "loss_max"), SCOPE, "2.989배"),
    Check("defaults.64k.flashinfer_tc_over_1.25", "64K 격자 혼합 길이: FlashInfer tensor-core 손실이 1.25배를 넘는 셀 수",
          12, 0, "", _defaults_summary("64k", "ragged", "flashinfer_tc", "over_1.25"), SCOPE, "12셀"),
    Check("defaults.64k.flashinfer_cc_max", "64K 격자 혼합 길이: FlashInfer CUDA-core 손실 최댓값",
          1.014, 0.0005, "x", _defaults_summary("64k", "ragged", "flashinfer_cc", "loss_max"), SCOPE, "1.014배"),
    Check("defaults.64k.fa2_best_max", "64K 격자 혼합 길이: FA2 분할만 고르는 선택기 손실 최댓값",
          1.049, 0.0005, "x", _defaults_summary("64k", "ragged", "fa2_best", "loss_max"), SCOPE, "1.049배"),
    Check("defaults.64k.uniform_fa2_heuristic_max", "64K 격자 균일 길이(2셀): FA2 휴리스틱 손실 최댓값",
          1.003, 0.0005, "x", _defaults_summary("64k", "uniform", "fa2_heuristic", "loss_max"), SCOPE, "1.003배"),

    Check("longctx.matched_cells_per_length", "긴 요청 길이별 비교: 두 격자가 공유하는 조건(B 26/32/64, 긴 요청 1~2개, 짧은 512~2048)의 셀 수",
          18, 0, "", lambda repo, data: float((_matched_long_cells(data).long == 65536).sum()), SCOPE, "길이마다 18셀"),
    Check("longctx.heuristic_median_8k", "긴 요청 8K: FA2 휴리스틱 / 최선 FA2 분할 손실 중앙값",
          1.545, 0.0005, "x", _by_long(8192, "fa2_loss", "median"), SCOPE, "1.545"),
    Check("longctx.heuristic_median_16k", "긴 요청 16K: 같은 손실 중앙값",
          1.933, 0.0005, "x", _by_long(16384, "fa2_loss", "median"), SCOPE, "1.933"),
    Check("longctx.heuristic_median_32k", "긴 요청 32K: 같은 손실 중앙값",
          2.558, 0.0005, "x", _by_long(32768, "fa2_loss", "median"), SCOPE, "2.558"),
    Check("longctx.heuristic_median_64k", "긴 요청 64K: 같은 손실 중앙값",
          2.839, 0.0005, "x", _by_long(65536, "fa2_loss", "median"), SCOPE, "2.839"),
    Check("longctx.heuristic_max_64k", "긴 요청 64K: 같은 손실 최댓값(65536+512×25, B=26)",
          5.183, 0.0005, "x", _by_long(65536, "fa2_loss", "max"), SCOPE, "5.183"),
    Check("longctx.flashinfer_tc_median_8k", "긴 요청 8K: FlashInfer tensor-core / 전체 최선 손실 중앙값",
          1.126, 0.0005, "x", _by_long(8192, "flashinfer_tc_loss", "median"), SCOPE, "1.126"),
    Check("longctx.flashinfer_tc_median_64k", "긴 요청 64K: FlashInfer tensor-core 손실 중앙값",
          1.634, 0.0005, "x", _by_long(65536, "flashinfer_tc_loss", "median"), SCOPE, "1.634"),
    Check("longctx.flashinfer_cc_max_all", "모든 긴 요청 길이: FlashInfer CUDA-core 손실 최댓값",
          1.014, 0.0005, "x", lambda repo, data: float(_matched_long_cells(data).flashinfer_cc_loss.max()), SCOPE, "1.014"),
    Check("longctx.headline_64k_heuristic_us", "64K 대표 셀(65536+512×31, B=32): FA2 휴리스틱 커널 시간",
          1956.961, 0.0005, "us", _headline_64k("fa2_heuristic_us"), SCOPE, "1957.0"),
    Check("longctx.headline_64k_best_fa2_us", "64K 대표 셀: 최선 FA2 분할(16) 커널 시간",
          421.873, 0.0005, "us", _headline_64k("fa2_best_us"), SCOPE, "421.9"),
    Check("longctx.headline_64k_flashinfer_cc_us", "64K 대표 셀: FlashInfer CUDA-core 커널 시간",
          414.703, 0.0005, "us", _headline_64k("flashinfer_cc_us"), SCOPE, "414.7"),
    Check("longctx.headline_64k_flashinfer_tc_us", "64K 대표 셀: FlashInfer tensor-core 커널 시간",
          1239.681, 0.0005, "us", _headline_64k("flashinfer_tc_us"), SCOPE, "1239.7"),

    Check("traffic.azure_conv_x1_requests", "Azure LLM 추론 트레이스(conv, 2023) 재생: 요청 수",
          19366, 0, "", _traffic("azure_conv_x1", "trace", "requests"), SCOPE, "19,366건"),
    Check("traffic.azure_conv_x1_steps", "Azure conv ×1(4090 설정: KV 10 GiB, 배치 64, step 30 ms): decode step 수",
          117035, 0, "", _traffic("azure_conv_x1", "steps"), SCOPE, "117,035 step"),
    Check("traffic.azure_conv_x1_batch_mean", "Azure conv ×1: 평균 배치 크기",
          34.77, 0.005, "", _traffic("azure_conv_x1", "losses", "batch_mean"), SCOPE, "평균 배치 34.8"),
    Check("traffic.azure_conv_x1_no_split_frac", "Azure conv ×1: 휴리스틱이 분할 없이 돌아가는(B ≥ 26) step 비율",
          0.895, 0.0005, "", _traffic("azure_conv_x1", "losses", "no_split_frac"), SCOPE, "89.5%"),
    Check("traffic.azure_conv_x1_loss_ge_1.1", "Azure conv ×1: 예측 손실 1.1배 이상인 step 비율",
          0.465, 0.0005, "", _traffic("azure_conv_x1", "losses", "loss_ge_1.1_frac"), SCOPE, "46.5%"),
    Check("traffic.azure_conv_x1_loss_ge_1.25", "Azure conv ×1: 예측 손실 1.25배 이상인 step 비율",
          0.135, 0.0005, "", _traffic("azure_conv_x1", "losses", "loss_ge_1.25_frac"), SCOPE, "13.5%"),
    Check("traffic.azure_conv_x1_loss_ge_1.5", "Azure conv ×1: 예측 손실 1.5배 이상인 step 비율",
          0.006, 0.0005, "", _traffic("azure_conv_x1", "losses", "loss_ge_1.5_frac"), SCOPE, "0.6%"),
    Check("traffic.azure_conv_x1_loss_p90", "Azure conv ×1: step 예측 손실의 90퍼센타일",
          1.278, 0.0005, "x", _traffic("azure_conv_x1", "losses", "loss_p90"), SCOPE, "1.278"),
    Check("traffic.azure_conv_x1_loss_max", "Azure conv ×1: step 예측 손실 최댓값",
          2.192, 0.0005, "x", _traffic("azure_conv_x1", "losses", "loss_max"), SCOPE, "2.192"),
    Check("traffic.azure_conv_x1_attention_ratio", "Azure conv ×1: 전체 step의 휴리스틱 attention 시간 합 / 최선 분할 시간 합",
          1.119, 0.0005, "x", _traffic("azure_conv_x1", "losses", "attention_time_ratio"), SCOPE, "1.119"),
    Check("traffic.azure_conv_x1_share_ge_1.25", "Azure conv ×1: 손실 1.25배 이상 step이 차지하는 attention 시간 비중",
          0.162, 0.0005, "", _traffic("azure_conv_x1", "losses", "attention_share_loss_ge_1.25"), SCOPE, "16.2%"),
    Check("traffic.azure_conv_x0.5_loss_ge_1.25", "Azure conv ×0.5(평균 배치 17): 손실 1.25배 이상 step 비율",
          0.024, 0.0005, "", _traffic("azure_conv_x0.5", "losses", "loss_ge_1.25_frac"), SCOPE, "2.4%"),
    Check("traffic.azure_conv_x2_loss_ge_1.25", "Azure conv ×2(평균 배치 47): 손실 1.25배 이상 step 비율",
          0.041, 0.0005, "", _traffic("azure_conv_x2", "losses", "loss_ge_1.25_frac"), SCOPE, "4.1%"),
    Check("traffic.azure_conv_x2_attention_ratio", "Azure conv ×2: attention 시간 비",
          1.092, 0.0005, "x", _traffic("azure_conv_x2", "losses", "attention_time_ratio"), SCOPE, "1.092"),
    Check("traffic.azure_conv_x1_kv436k_loss_ge_1.25", "Azure conv ×1, KV 436K 토큰(80 GB급 가정): 손실 1.25배 이상 step 비율(KV 예산이 제약이 아님)",
          0.137, 0.0005, "", _traffic("azure_conv_x1_kv436k", "losses", "loss_ge_1.25_frac"), SCOPE, "13.7%"),
    Check("traffic.azure_conv_x1_prompt2_loss_ge_1.25", "what-if 프롬프트 ×2(Azure conv ×1, KV 10 GiB): 손실 1.25배 이상 step 비율",
          0.257, 0.0005, "", _traffic("azure_conv_x1_prompt2", "losses", "loss_ge_1.25_frac"), SCOPE, "25.7%"),
    Check("traffic.azure_conv_x1_prompt2_attention_ratio", "what-if 프롬프트 ×2: attention 시간 비",
          1.156, 0.0005, "x", _traffic("azure_conv_x1_prompt2", "losses", "attention_time_ratio"), SCOPE, "1.156"),
    Check("traffic.azure_conv_x1_prompt4_batch_mean", "what-if 프롬프트 ×4(KV 10 GiB): 평균 배치가 분할 임계(26) 아래로 떨어짐",
          14.56, 0.005, "", _traffic("azure_conv_x1_prompt4", "losses", "batch_mean"), SCOPE, "14.6"),
    Check("traffic.azure_conv_x1_prompt4_loss_ge_1.25", "what-if 프롬프트 ×4(KV 10 GiB): 손실 1.25배 이상 step 비율",
          0.029, 0.0005, "", _traffic("azure_conv_x1_prompt4", "losses", "loss_ge_1.25_frac"), SCOPE, "2.9%"),
    Check("traffic.azure_code_x16_loss_ge_1.25", "Azure code ×16(평균 배치 27): 손실 1.25배 이상 step 비율",
          0.141, 0.0005, "", _traffic("azure_code_x16", "losses", "loss_ge_1.25_frac"), SCOPE, "14.1%"),
    Check("traffic.azure_code_x16_attention_ratio", "Azure code ×16: attention 시간 비",
          1.136, 0.0005, "x", _traffic("azure_code_x16", "losses", "attention_time_ratio"), SCOPE, "1.136"),
    Check("traffic.burstgpt_x100_loss_ge_1.25", "BurstGPT 첫 7일 ×100(평균 배치 27, 프롬프트 중앙값 502): 손실 1.25배 이상 step 비율",
          0.003, 0.0005, "", _traffic("burstgpt_7d_x100", "losses", "loss_ge_1.25_frac"), SCOPE, "0.3%"),
    Check("traffic.burstgpt_x100_attention_ratio", "BurstGPT ×100: attention 시간 비",
          1.018, 0.0005, "x", _traffic("burstgpt_7d_x100", "losses", "attention_time_ratio"), SCOPE, "1.018"),
    Check("traffic.burstgpt_gpt4_x300_loss_ge_1.25", "BurstGPT GPT-4 14일 ×300(평균 배치 30): 손실 1.25배 이상 step 비율",
          0.015, 0.0005, "", _traffic("burstgpt_gpt4_14d_x300", "losses", "loss_ge_1.25_frac"), SCOPE, "1.5%"),
    Check("traffic.azure_conv_x1_kv436k_prompt2_loss_ge_1.25", "what-if 프롬프트 ×2, KV 436K 토큰(배치 34.8 유지): 손실 1.25배 이상 step 비율",
          0.293, 0.0005, "", _traffic("azure_conv_x1_kv436k_prompt2", "losses", "loss_ge_1.25_frac"), SCOPE, "29.3%"),
    Check("traffic.azure_conv_x1_kv436k_prompt2_attention_ratio", "what-if 프롬프트 ×2, KV 436K: attention 시간 비",
          1.179, 0.0005, "x", _traffic("azure_conv_x1_kv436k_prompt2", "losses", "attention_time_ratio"), SCOPE, "1.179"),
    Check("traffic.azure_conv_x1_kv436k_prompt4_loss_ge_1.25", "what-if 프롬프트 ×4, KV 436K 토큰: 손실 1.25배 이상 step 비율",
          0.395, 0.0005, "", _traffic("azure_conv_x1_kv436k_prompt4", "losses", "loss_ge_1.25_frac"), SCOPE, "39.5%"),
    Check("traffic.azure_conv_x1_kv436k_prompt4_attention_ratio", "what-if 프롬프트 ×4, KV 436K: attention 시간 비",
          1.222, 0.0005, "x", _traffic("azure_conv_x1_kv436k_prompt4", "losses", "attention_time_ratio"), SCOPE, "1.222"),

    Check("serve64k.heuristic_tpot", "Qwen3-4B 64K 혼합 길이(65536×1 + 512×31, 64토큰, 캐시 유지 규약, 3회): 기본 휴리스틱 TPOT",
          96.93, 0.005, "ms", _longctx_serve("heuristic", "tpot_ms_mean"), SCOPE, "96.93"),
    Check("serve64k.heuristic_attn", "64K 혼합 길이: step당 attention 시간(휴리스틱)",
          71.70, 0.005, "ms", _longctx_serve("heuristic", "attn_ms_per_step"), SCOPE, "71.70"),
    Check("serve64k.heuristic_attn_share", "64K 혼합 길이: decode step에서 attention 비중(휴리스틱)",
          0.828, 0.0005, "", _longctx_serve("heuristic", "attn_share"), SCOPE, "82.8%"),
    Check("serve64k.table_tpot", "64K 혼합 길이: 측정 테이블 정책 TPOT(32K 셀에서 외삽한 분할 8)",
          40.06, 0.005, "ms", _longctx_serve("table", "tpot_ms_mean"), SCOPE, "40.06"),
    Check("serve64k.table_speedup", "64K 혼합 길이: 측정 테이블 정책의 TPOT 개선 배율",
          2.419, 0.0005, "x", _longctx_serve("table", "speedup_vs_heuristic"), SCOPE, "2.419배"),
    Check("serve64k.model_speedup", "64K 혼합 길이: 성능 모델 정책의 TPOT 개선 배율",
          2.463, 0.0005, "x", _longctx_serve("model", "speedup_vs_heuristic"), SCOPE, "2.463배"),
    Check("serve64k.hybrid_speedup", "64K 혼합 길이: 혼합 정책의 TPOT 개선 배율",
          2.463, 0.0005, "x", _longctx_serve("hybrid", "speedup_vs_heuristic"), SCOPE, "2.463배"),
    Check("serve64k.hybrid_attn", "64K 혼합 길이: step당 attention 시간(혼합 정책)",
          14.55, 0.005, "ms", _longctx_serve("hybrid", "attn_ms_per_step"), SCOPE, "14.55"),
    Check("serve64k.hybrid_attn_share", "64K 혼합 길이: attention 비중(혼합 정책)",
          0.496, 0.0005, "", _longctx_serve("hybrid", "attn_share"), SCOPE, "49.6%"),
    Check("serve64k.policies_choose_split8_every_step", "64K 혼합 길이: 테이블·모델·혼합 정책 모두 63 step 전부 분할 8(1종류×1000 + 8)",
          1008, 0, "", _longctx_splits("hybrid"), SCOPE, "63 step 전부 분할 8"),
    Check("serve64k.table_splits", "64K 혼합 길이: 테이블 정책의 분할 수(같은 값)",
          1008, 0, "", _longctx_splits("table")),
    Check("serve64k.divergence_events_per_repeat", "64K 혼합 길이: 휴리스틱 대비 분기 사건 수(정책마다 반복당 최대)",
          1, 0, "", _longctx_divergence("hybrid", "events_repeat_max"), SCOPE, "분기 사건 1건"),
    Check("serve64k.divergence_token_mismatches", "64K 혼합 길이: 2,048 토큰 중 다른 토큰 수(반복당 최대)",
          4, 0, "", _longctx_divergence("hybrid", "token_mismatches_max"), SCOPE, "2,048 토큰 중 4개"),
    Check("serve64k.divergence_same_position_every_repeat", "64K 혼합 길이: 세 정책·세 반복 모두 같은 자리(64K 요청, 위치 27)에서 갈림(1=참)",
          1, 0, "", lambda repo, data: float(all(_longctx_divergence(p, "same_positions_every_repeat")(repo, data)
                                                   for p in ("table", "model", "hybrid"))), SCOPE, "위치 27"),
    Check("serve64k.divergence_clear", "64K 혼합 길이: 교사 강제 진단으로 분류한 분기 사건 중 clear(참조 로짓 여유가 bf16 간격 2개 초과) 수",
          1, 0, "", _longctx_divergence("hybrid", "clear"), SCOPE, "clear 1건"),
    Check("serve64k.divergence_margin_ulps", "64K 혼합 길이: 그 사건에서 참조 로짓의 후보 토큰까지 여유(bf16 간격 단위)",
          11.0, 0.05, "ulp", lambda repo, data: float(pd.read_csv(data / "serve_4090" / LONGCTX_RUN / "divergence.csv")
                                                      .query("policy == 'hybrid' and events > 0").margin_ulps.iloc[0]), SCOPE, "11 ulp"),
    Check("serve64k.divergence_reproduced", "64K 혼합 길이: 교사 강제 실행이 자유 생성과 같은 토큰을 냈는가(1=재현)",
          1, 0, "", lambda repo, data: float(pd.read_csv(data / "serve_4090" / LONGCTX_RUN / "divergence.csv")
                                            .query("policy == 'hybrid' and events > 0").teacher_reproduced.all()), SCOPE, "재현"),
    Check("serve64k.probe_kernel_err", "64K clear 사건의 커널 탐침(같은 KV 상태, 36층, fp32 참조, 2026-10-08): 분할 1·8 attention 출력의 참조 대비 최대 오차(출력 최대 31.8인 층; bf16 간격 0.125)",
          0.123, 0.0005, "", _probe64k_kernel_err, SCOPE, "최대 0.123"),
    Check("serve64k.probe_split_diff", "64K 탐침: 분할 1 vs 8 층별 출력 차이 최대(그 층 출력 크기의 bf16 간격 1개)",
          0.125, 0.0005, "", _probe64k_split_diff, SCOPE, "최대 0.125"),
    Check("serve64k.probe_layers_over_one_ulp", "64K 탐침: 분할 차이가 그 층 최대 출력의 bf16 간격 1개를 넘는 층 수(0 = 구현 오류 아님)",
          0, 0, "", _probe64k_layers_over_one_ulp, SCOPE, "기준 넘는 층 0개"),
    Check("serve64k.probe_single_step_argmax_changes", "64K 탐침: 같은 KV 상태에서 분할 8로 계산한 단일 step에서 argmax가 바뀐 요청 수(64K 요청 하나)",
          1, 0, "", _probe64k_argmax_changes, SCOPE, "argmax가 바뀐다"),
    Check("serve64k.probe_single_step_dlogit", "64K 탐침: 그 요청의 단일 step 최대 |Δlogit|",
          8.99, 0.005, "", _probe64k_target("max_abs_logit_diff"), SCOPE, "8.99"),
    Check("serve64k.probe_single_step_cosine", "64K 탐침: 그 요청의 두 로짓 벡터 코사인 유사도",
          0.100, 0.0005, "", _probe64k_target("cosine"), SCOPE, "코사인 0.10"),
    Check("serve64k.probe_short_requests_identical", "64K 탐침: 짧은 요청 31개의 단일 step 로짓 차이 최대(0 = 두 분할이 완전히 같음)",
          0, 0, "", _probe64k_short_requests_dlogit, SCOPE, "차이 0)"),

    Check("fiengine.repeats", "엔진 안 FlashInfer 캠페인: 정책마다 반복 횟수(세 시나리오·네 정책 공통)",
          3, 0, "", _fiengine_repeats, SCOPE, "반복 3회"),
    Check("fiengine.kv_gib", "엔진 안 FlashInfer 캠페인: KV 페이지 풀 예산(세 시나리오 공통)",
          10, 0.0005, "GiB", _fiengine_manifest_kv_gib, SCOPE, "KV 10 GiB"),
    *_fiengine_generated_checks(),
    Check("fiengine.table_token_mismatches", "엔진 안 FlashInfer 캠페인: table 정책의 불일치 토큰 수(세 시나리오 합; 0 = 휴리스틱과 토큰 동일)",
          0, 0, "", _fiengine_table_policy_sum("token_mismatches_max"), SCOPE, "table 정책은 세 시나리오 모두 불일치 토큰 0개"),
    Check("fiengine.table_events", "엔진 안 FlashInfer 캠페인: table 정책의 분기 사건 수(세 시나리오 합)",
          0, 0, "", _fiengine_table_policy_sum("distinct_positions"), SCOPE, "분기 사건 0건"),
    Check("fiengine.same_positions_every_repeat", "엔진 안 FlashInfer 캠페인: 세 시나리오 × FlashInfer 두 정책 여섯 쌍 모두 세 반복에서 사건의 자리가 같음(1=참)",
          1, 0, "", _fiengine_same_positions_every_repeat, SCOPE, "세 반복에서 사건의 자리가 같다"),
    Check("fiengine.nothing_unclassified", "엔진 안 FlashInfer 캠페인: 미분류·미재현·토큰 누락·사건 밖 불일치의 합(모든 정책·시나리오)",
          0, 0, "", _fiengine_needs_investigation, SCOPE, "미분류·미재현·토큰 누락·사건 밖 불일치는 모두 0"),
    Check("fiengine.clear_position", "혼합 길이의 clear 사건: 요청 번호×1000 + 처음 갈린 위치(두 FlashInfer 정책이 같은 자리, 8041 = 요청 8 위치 41)",
          8041, 0, "", _fiengine_clear_position, SCOPE, "요청 8 위치 41"),
    Check("fiengine.clear_position_everywhere", "혼합·균일 길이 두 시나리오·두 정책의 clear 사건이 모두 같은 자리이고 도착 시나리오에는 clear가 없음",
          8041, 0, "", _fiengine_clear_position_anywhere, SCOPE, "clear는 이 한 자리뿐"),
    Check("fiengine.clear_reference_token", "혼합 길이 clear 사건: 참조(FA2 휴리스틱) 토큰",
          25, 0, "", _fiengine_clear_value("reference_token"), SCOPE, "참조 토큰 25"),
    Check("fiengine.clear_candidate_token", "혼합 길이 clear 사건: 후보(FlashInfer) 토큰",
          198, 0, "", _fiengine_clear_value("candidate_token"), SCOPE, "후보 토큰 198"),
    Check("fiengine.clear_top_logit", "혼합 길이 clear 사건: 참조의 1위 로짓",
          33.75, 0.005, "", _fiengine_clear_value("reference_top1_logit"), SCOPE, "1위 로짓 33.75"),
    Check("fiengine.clear_margin", "혼합 길이 clear 사건: 참조 로짓에서 후보 토큰까지의 여유",
          17.5, 0.005, "", _fiengine_clear_value("reference_margin"), SCOPE, "여유 17.5"),
    Check("fiengine.clear_margin_ulps", "혼합 길이 clear 사건: 그 여유(bf16 간격 단위)",
          70, 0.05, "ulp", _fiengine_clear_value("margin_ulps"), SCOPE, "= 70 ulp"),
    Check("fiengine.clear_dlogit_ragged_tensor_core", "혼합 길이 clear 사건 위치의 교사 강제 로짓 최대 차이(FlashInfer tensor-core)",
          22.9, 0.05, "", _fiengine_clear_dlogit("ragged", "flashinfer"), SCOPE, "혼합 길이 tensor-core 22.9"),
    Check("fiengine.clear_dlogit_ragged_cuda_core", "같은 위치의 로짓 최대 차이(혼합 길이, FlashInfer CUDA-core)",
          22.3, 0.05, "", _fiengine_clear_dlogit("ragged", "flashinfer_cudacore"), SCOPE, "혼합 길이 tensor-core 22.9 · CUDA-core 22.3"),
    Check("fiengine.clear_dlogit_uniform_tensor_core", "균일 길이 clear 사건 위치의 로짓 최대 차이(FlashInfer tensor-core)",
          22.9, 0.05, "", _fiengine_clear_dlogit("uniform", "flashinfer"), SCOPE, "균일 길이 tensor-core 22.9"),
    Check("fiengine.clear_dlogit_uniform_cuda_core", "같은 위치의 로짓 최대 차이(균일 길이, FlashInfer CUDA-core)",
          21.6, 0.05, "", _fiengine_clear_dlogit("uniform", "flashinfer_cudacore"), SCOPE, "균일 길이 tensor-core 22.9 · CUDA-core 21.6"),
    Check("fiengine.clear_is_the_control_event", "그 clear 사건 = 10월 2일 대조군(fixed:8)의 clear 사건: 같은 요청·위치·두 토큰·여유(1=같음)",
          1, 0, "", _fiengine_clear_matches_control, SCOPE, "10월 2일 대조군의 `clear`와 같은 사건"),
    Check("fiengine.rid8_prompt_same", "요청 8의 합성 프롬프트(길이·해시)가 세 시나리오와 10월 2일 대조군에서 같음(1=같음)",
          1, 0, "", _fiengine_rid8_prompt_same_as_control, SCOPE, "요청 8의 합성 프롬프트는 같다"),
    *_anytable_generated_checks(),
    Check("anytable.repeats", "라이브러리 무관 표 캠페인: 정책마다 반복 횟수(세 시나리오·네 정책 공통)",
          3, 0, "", _anytable_repeats, SCOPE, "warm-up 뒤 반복 3회"),
    Check("anytable.kv_gib", "라이브러리 무관 표 캠페인: KV 페이지 풀 예산(세 시나리오 공통)",
          10, 0.0005, "GiB", _anytable_kv_gib, SCOPE, "반복 3회, KV 10 GiB"),
    Check("anytable.run_start", "라이브러리 무관 표 캠페인: 첫 실행 시작 시각(KST, 시·분을 hhmm으로)",
          1644, 0, "", _anytable_run_time("start"), SCOPE, "2026-10-08 16:44~"),
    Check("anytable.run_end", "라이브러리 무관 표 캠페인: 마지막 실행 종료 시각(KST, hhmm)",
          1655, 0, "", _anytable_run_time("end"), SCOPE, "16:55)"),
    Check("anytable.code_commit", "라이브러리 무관 표 캠페인: 실행 코드가 커밋 b8e1707에 미커밋 변경이 얹힌 상태(1=참)",
          1, 0, "", _anytable_code_state, SCOPE, "코드는 커밋 b8e1707에 미커밋 변경이 얹힌 상태이고"),
    Check("anytable.steps_per_repeat", "라이브러리 무관 표 캠페인: 반복당 decode step 수(세 시나리오·네 정책 공통)",
          63, 0, "", _anytable_steps_per_repeat, SCOPE, "반복당 63 step이고 세 반복이 같다"),
    Check("anytable.n_layers", "라이브러리 무관 표 캠페인: 모델의 층 수(기록된 model_config)",
          36, 0, "", _anytable_n_layers, SCOPE, "Qwen3-4B는 36층이라"),
    # the table (demo_data/dispatch_paged_cold_any.csv)
    Check("anytable.table_is_the_grid_table", "표 = 번들의 여섯 32K 격자에서 static_default_losses로 다시 계산한 완전한 셀(1=같음)",
          1, 0, "", _anytable_table_is_the_grid_table, SCOPE, "번들의 격자에서 다시 계산한 표와 같다"),
    Check("anytable.table_cells", "라이브러리 무관 표: 세 변형을 모두 잰 셀 수",
          211, 0, "", _anytable_cell_count, SCOPE, "211셀"),
    Check("anytable.table_best_cudacore", "라이브러리 무관 표: 전체 최선이 FlashInfer CUDA-core인 셀 수",
          148, 0, "", _anytable_best_count(("flashinfer_paged_cudacore",)), SCOPE, "CUDA-core 148셀"),
    Check("anytable.table_best_tensorcore", "라이브러리 무관 표: 전체 최선이 FlashInfer tensor-core인 셀 수",
          45, 0, "", _anytable_best_count(("flashinfer_paged",)), SCOPE, "tensor-core 45셀"),
    Check("anytable.table_best_fa2", "라이브러리 무관 표: 전체 최선이 FA2 분할 변형인 셀 수",
          18, 0, "", _anytable_best_count(None), SCOPE, "FA2 분할 18셀"),
    Check("anytable.table_ragged_cells", "라이브러리 무관 표: 혼합 길이 셀 수",
          162, 0, "", _anytable_ragged_cells, SCOPE, "혼합 길이 162셀"),
    Check("anytable.table_ragged_best_cudacore", "라이브러리 무관 표: 혼합 길이 셀 중 전체 최선이 CUDA-core인 셀 수",
          139, 0, "", _anytable_best_count(("flashinfer_paged_cudacore",), ragged_only=True), SCOPE, "CUDA-core 139셀"),
    Check("anytable.table_ragged_best_tensorcore", "라이브러리 무관 표: 혼합 길이 셀 중 전체 최선이 tensor-core인 셀 수",
          21, 0, "", _anytable_best_count(("flashinfer_paged",), ragged_only=True), SCOPE, "tensor-core 21셀"),
    Check("anytable.table_ragged_best_fa2", "라이브러리 무관 표: 혼합 길이 셀 중 전체 최선이 FA2 분할 변형인 셀 수",
          2, 0, "", _anytable_best_count(None, ragged_only=True), SCOPE, "FA2 2셀"),
    Check("anytable.cudacore_gain_median", "혼합 길이에서 CUDA-core가 최선인 셀: 최선 FA2 분할보다 빠른 폭의 중앙값(호출당)",
          9.8, 0.05, "us", _anytable_cudacore_gain("median"), SCOPE, "커널 호출당 중앙값 9.8 µs"),
    Check("anytable.cudacore_gain_max", "혼합 길이에서 CUDA-core가 최선인 셀: 그 폭의 최댓값(호출당)",
          39, 0.5, "us", _anytable_cudacore_gain("max"), SCOPE, "최대 39 µs"),
    # plan cost
    Check("anytable.plan_us_spec", "table_any_p371 정책 스펙에 기록된 FlashInfer 후보의 step당 plan 비용",
          371, 0, "us", _anytable_plan_us_spec, SCOPE, "FlashInfer 후보에는 step마다 `plan()` 시간 371 µs를 더한다"),
    Check("anytable.plan_us_default", "TablePolicy의 plan_us 기본값(0 = 커널 시간만 보는 전체 최선, TODO의 정의)",
          0, 0, "us", _anytable_plan_us_default, SCOPE, "기본값 0은 TODO의 정의"),
    Check("anytable.plan_us_measured", "그 값의 출처: 10월 8일 FlashInfer 캠페인 혼합 길이 CUDA-core 정책의 step당 plan() 시간",
          370.8, 0.05, "us", _anytable_fiengine_serve("ragged", "flashinfer_cudacore", "policy_us_per_step"), SCOPE,
          "step당 `plan()` 370.8 µs를 반올림한 값"),
    # the cells the scenarios read
    Check("anytable.ragged_cell_fa2_splits", "혼합 길이 실생성이 읽은 셀: 최선 FA2 분할 수",
          16, 0, "", _anytable_cell_fa2_splits("ragged"), SCOPE, "최선 FA2 분할(16)"),
    Check("anytable.ragged_cell_fa2_us", "혼합 길이 실생성이 읽은 셀: 최선 FA2 분할의 커널 시간",
          279.4, 0.05, "us", _anytable_cell_value("ragged", "fa2_best_us"), SCOPE, "분할(16)이 279.4 µs"),
    Check("anytable.ragged_cell_cudacore_us", "같은 셀: CUDA-core 커널 시간",
          269.9, 0.05, "us", _anytable_cell_value("ragged", "flashinfer_cc_us"), SCOPE, "CUDA-core가 269.9 µs"),
    Check("anytable.ragged_cell_gain_per_call", "같은 셀: CUDA-core가 최선 FA2 분할보다 빠른 폭(호출당)",
          9.4, 0.05, "us", _anytable_cell_gain_us(False), SCOPE, "호출당 9.4 µs 차이"),
    Check("anytable.ragged_cell_gain_per_step", "같은 셀: 그 폭 × 층 수 = step당 커널 이득",
          340, 0.5, "us", _anytable_cell_gain_us(True), SCOPE, "step당 340 µs다"),
    Check("anytable.ragged_cell_cost_gap", "같은 셀: plan 비용 371 µs를 더한 CUDA-core 비용 − FA2 비용(양수 = 비용 모델은 FA2가 쌈)",
          31, 0.5, "us", _anytable_cell_cost_gap_us, SCOPE, "FA2 쪽을 step당 31 µs 싸게 본다"),
    Check("anytable.uniform_cell_batch", "균일 길이 실생성이 읽은 셀: 배치 크기",
          32, 0, "", _anytable_cell_value("uniform", "B"), SCOPE, "(B=32,"),
    Check("anytable.uniform_cell_length", "균일 길이 실생성이 읽은 셀: 요청 길이",
          512, 0, "", _anytable_cell_value("uniform", "L_kv"), SCOPE, ", 길이 512)"),
    Check("anytable.uniform_cell_fa2_us", "균일 길이 셀: 최선 FA2(= FA2 휴리스틱)의 커널 시간",
          103.6, 0.05, "us", _anytable_cell_value("uniform", "fa2_best_us"), SCOPE, "FA2 103.6 µs"),
    Check("anytable.uniform_cell_tensorcore_us", "균일 길이 셀: FlashInfer tensor-core 커널 시간",
          104.2, 0.05, "us", _anytable_cell_value("uniform", "flashinfer_tc_us"), SCOPE, "tensor-core 104.2 µs"),
    Check("anytable.uniform_cell_cudacore_us", "균일 길이 셀: FlashInfer CUDA-core 커널 시간",
          105.6, 0.05, "us", _anytable_cell_value("uniform", "flashinfer_cc_us"), SCOPE, "CUDA-core 105.6 µs"),
    Check("anytable.replay_differences", "기록한 step 길이로 세 table 정책을 CPU에서 다시 돌렸을 때 기록된 (백엔드, 분할)과 다른 step 수",
          0, 0, "", _anytable_replay_differences, SCOPE, "기록된 백엔드·분할과 다른 step이 0개다"),
    # reading the numbers
    Check("anytable.ragged_fiengine_cudacore_tpot", "10월 8일 FlashInfer 캠페인 혼합 길이: 고정 CUDA-core 정책의 TPOT",
          33.28, 0.005, "ms", _anytable_fiengine_serve("ragged", "flashinfer_cudacore", "tpot_ms_mean"), SCOPE,
          "§4의 CUDA-core 정책(33.28 ms"),
    Check("anytable.ragged_fiengine_cudacore_speedup", "같은 정책의 TPOT 개선 배율",
          1.833, 0.0005, "x", _anytable_fiengine_serve("ragged", "flashinfer_cudacore", "speedup_vs_heuristic"), SCOPE,
          "33.28 ms, 1.833배)"),
    Check("anytable.ragged_table_any_vs_table_pct", "혼합 길이: table_any의 TPOT가 table보다 짧은 정도",
          1.15, 0.005, "%", _anytable_tpot_pct("ragged", "table", "table_any"), SCOPE, "FA2 `table`보다 1.15% 짧다"),
    Check("anytable.ragged_attn_gain_ms", "혼합 길이: table_any의 attention 시간이 table보다 짧은 정도",
          0.54, 0.005, "ms", _anytable_gap("ragged", "table", "table_any", "attn_ms_per_step"), SCOPE,
          "`table`보다 0.54 ms/step 짧은데"),
    Check("anytable.ragged_tpot_gain_ms", "혼합 길이: table_any의 TPOT가 table보다 짧은 정도",
          0.38, 0.005, "ms", _anytable_gap("ragged", "table", "table_any", "tpot_ms_mean"), SCOPE, "TPOT 이득 0.38 ms는"),
    Check("anytable.ragged_attn_gain_minus_plan_ms", "혼합 길이: attention 절감 − table_any의 정책 비용(plan 포함)",
          0.16, 0.005, "ms", _anytable_attn_gain_minus_plan_ms("ragged"), SCOPE, "`plan()` 시간을 뺀 값 0.16 ms보다 크다"),
    Check("anytable.ragged_table_any_faster_repeats", "혼합 길이: table_any의 TPOT가 table보다 짧았던 반복 수(3 = 모두)",
          3, 0, "", _anytable_faster_repeats("ragged"), SCOPE, "세 반복 모두 `table_any`가 `table`보다 빨랐다"),
    Check("anytable.ragged_p371_splits16_every_step", "혼합 길이: table_any_p371이 FA2 분할 16으로 돈 step 수(= 반복당 step 수)",
          63, 0, "", _anytable_split_steps("ragged", "table_any_p371", 16), SCOPE, "모든 step에서 FA2 분할 16을 골랐고"),
    Check("anytable.ragged_p371_vs_table_pct", "혼합 길이: table_any_p371의 TPOT가 table보다 긴 정도(같은 커널·분할 선택)",
          0.21, 0.005, "%", _anytable_tpot_pct("ragged", "table_any_p371", "table"), SCOPE, "TPOT는 `table`보다 0.21% 길 뿐인데"),
    Check("anytable.ragged_attn_gain_per_layer", "혼합 길이: 엔진 안에서 table_any의 attention 절감을 층 수로 나눈 값",
          15.1, 0.05, "us", _anytable_attn_gain_per_layer_us("ragged"), SCOPE, "층당 15.1 µs"),
    Check("anytable.ragged_spread", "혼합 길이: 세 table 정책의 TPOT 최대/최소 − 1",
          1.36, 0.005, "%", _anytable_spread_pct("ragged"), SCOPE, "세 정책의 TPOT 차이는 1.36% 이내다"),
    Check("anytable.uniform_spread", "균일 길이: 세 table 정책의 TPOT 최대/최소 − 1",
          0.07, 0.005, "%", _anytable_spread_pct("uniform"), SCOPE, "세 정책의 TPOT 차이는 0.07% 이내다"),
    Check("anytable.arrivals_table_any_vs_table_pct", "요청 도착: table_any의 TPOT가 table보다 짧은 정도",
          0.36, 0.005, "%", _anytable_tpot_pct("arrivals", "table", "table_any"), SCOPE, "`table`보다 0.36% 빠르고"),
    Check("anytable.arrivals_table_any_faster_repeats", "요청 도착: table_any의 TPOT가 table보다 짧았던 반복 수(3 = 모두)",
          3, 0, "", _anytable_faster_repeats("arrivals"), SCOPE, "세 반복 모두 앞섰지만"),
    Check("anytable.arrivals_p371_vs_table_pct", "요청 도착: table_any_p371의 TPOT가 table보다 긴 정도(같은 선택)",
          0.19, 0.005, "%", _anytable_tpot_pct("arrivals", "table_any_p371", "table"), SCOPE, "`table`의 차이가 0.19%이므로"),
    Check("anytable.arrivals_spread", "요청 도착: 세 table 정책의 TPOT 최대/최소 − 1",
          0.55, 0.005, "%", _anytable_spread_pct("arrivals"), SCOPE, "세 정책의 TPOT 차이는 0.55% 이내다"),
    Check("anytable.p371_same_choices_as_table", "table_any_p371이 table과 (백엔드, 분할)이 다르게 돈 step 수(세 시나리오·세 반복 합)",
          0, 0, "", _anytable_p371_choice_differences, SCOPE, "`table_any_p371`은 `table`과 step마다 같은 분할을 골라"),
    # output identity and the divergence events
    Check("anytable.ragged_table_any_tokens_equal_fiengine", "혼합 길이: table_any와 10월 8일 캠페인 flashinfer_cudacore의 토큰 이력 차이(세 반복 합)",
          0, 0, "", _anytable_token_differences("ragged", "table_any", FIENGINE_RUN, "flashinfer_cudacore"), SCOPE,
          "§4의 `flashinfer_cudacore` 정책과 반복 3회 모두 한 토큰도 다르지 않고"),
    Check("anytable.arrivals_table_any_tokens_equal_fiengine", "요청 도착: table_any와 10월 8일 캠페인 flashinfer_cudacore의 토큰 이력 차이(세 반복 합)",
          0, 0, "", _anytable_token_differences("arrivals", "table_any", FIENGINE_RUN, "flashinfer_cudacore"), SCOPE,
          "§4의 `flashinfer_cudacore` 정책과 반복 3회 모두 한 토큰도 다르지 않고"),
    Check("anytable.events_equal_fiengine", "혼합 길이·요청 도착: table_any의 분기 사건이 flashinfer_cudacore의 것과 같음(자리·토큰·분류·여유·로짓 차이, 1=같음)",
          1, 0, "", _anytable_events_equal_fiengine, SCOPE, "분기 사건의 자리·토큰·분류도 같다"),
    Check("anytable.p371_tokens_equal_table", "table_any_p371과 table의 토큰 이력 차이(세 시나리오·세 반복 합)",
          0, 0, "", _anytable_p371_token_differences, SCOPE, "`table_any_p371`의 토큰 이력은 세 시나리오에서 `table`과 한 토큰도 다르지 않다"),
    Check("anytable.zero_policies_token_mismatches", "table·table_any_p371의 불일치 토큰 수(세 시나리오 합; 0 = 휴리스틱과 토큰 동일)",
          0, 0, "", _anytable_zero_policies("token_mismatches_max"), SCOPE, "세 시나리오 모두 불일치 토큰 0개"),
    Check("anytable.zero_policies_events", "table·table_any_p371의 분기 사건 수(세 시나리오 합)",
          0, 0, "", _anytable_zero_policies("distinct_positions"), SCOPE, "분기 사건 0건이다"),
    Check("anytable.same_positions_every_repeat", "세 시나리오 × table 정책 셋 아홉 쌍 모두 세 반복에서 사건의 자리가 같음(1=참)",
          1, 0, "", _anytable_same_positions_every_repeat, SCOPE, "사건의 자리는 세 반복에서 같다"),
    Check("anytable.nothing_unclassified", "미분류·미재현·토큰 누락·사건 밖 불일치의 합(모든 정책·시나리오)",
          0, 0, "", _anytable_needs_investigation, SCOPE, "미분류·미재현·토큰 누락·사건 밖 불일치는 모두 0이고"),
    Check("anytable.clear_is_the_fiengine_event", "혼합 길이 table_any의 clear 사건 = 10월 8일 캠페인 flashinfer_cudacore의 clear 사건(요청·위치·두 토큰·여유·로짓 차이, 1=같음)",
          1, 0, "", _anytable_clear_is_the_fiengine_event, SCOPE, "혼합 길이의 clear 한 자리는 §4b의 그 사건이다"),
    Check("anytable.clear_position", "혼합 길이 table_any의 clear 사건: 요청 번호×1000 + 처음 갈린 위치(8041 = 요청 8 위치 41)",
          8041, 0, "", _anytable_clear_position, SCOPE, "요청 8 위치 41"),
    Check("anytable.clear_margin_ulps", "같은 clear 사건: 참조 로짓에서 후보 토큰까지의 여유(bf16 간격 단위)",
          70, 0.05, "ulp", _anytable_clear_value("margin_ulps"), SCOPE, "여유 70 ulp"),
    Check("anytable.clear_dlogit", "같은 clear 사건 위치의 교사 강제 로짓 최대 차이",
          22.3, 0.05, "", _anytable_clear_value("max_abs_logit_diff"), SCOPE, "교사 강제 로짓 최대 차이 22.3"),

    Check("fixedsplit.s_star", "32K 혼합+균일 합집합(cold, 211셀)에서 최악 손실을 최소화하는 단일 고정 분할 수(최선 FA2 분할 대비)", 8, 0, "", _fixed_s_star, FIXEDSPLIT, "s*=8"),
    Check("fixedsplit.32k.ragged.fixed8_max", "32K 혼합 길이 162셀: 분할 8 고정 / 최선 FA2 분할 regret 최댓값", 1.042, 0.0005, "x", _fixed("32k_ragged", "fixed:8", "loss_max"), FIXEDSPLIT, "1.042배"),
    Check("fixedsplit.32k.ragged.fixed8_median", "32K 혼합 길이: 분할 8 고정 regret 중앙값", 1.01, 0.0005, "x", _fixed("32k_ragged", "fixed:8", "loss_median"), FIXEDSPLIT, "1.010배"),
    Check("fixedsplit.32k.ragged.fixed16_max", "32K 혼합 길이: 분할 16 고정 regret 최댓값", 1.069, 0.0005, "x", _fixed("32k_ragged", "fixed:16", "loss_max"), FIXEDSPLIT, "1.069배"),
    Check("fixedsplit.32k.ragged.fixed1_median", "32K 혼합 길이: 분할 1(분할 없음) 고정 regret 중앙값", 1.639, 0.0005, "x", _fixed("32k_ragged", "fixed:1", "loss_median"), FIXEDSPLIT, "1.639배"),
    Check("fixedsplit.32k.ragged.heuristic_median", "32K 혼합 길이: FA2 휴리스틱 / 최선 FA2 분할 regret 중앙값", 1.436, 0.0005, "x", _fixed("32k_ragged", "fa2_heuristic", "loss_median"), FIXEDSPLIT, "1.436배"),
    Check("fixedsplit.32k.ragged.heuristic_over1_25", "32K 혼합 길이: FA2 휴리스틱 regret이 1.25배를 넘는 셀 수(최선 FA2 분할 대비)", 108, 0, "", _fixed("32k_ragged", "fa2_heuristic", "over_1.25"), FIXEDSPLIT, "108셀"),
    Check("fixedsplit.32k.uniform.fixed8_max", "32K 균일 길이 49셀: 분할 8 고정 regret 최댓값", 1.127, 0.0005, "x", _fixed("32k_uniform", "fixed:8", "loss_max"), FIXEDSPLIT, "1.127배"),
    Check("fixedsplit.32k.uniform.fixed16_max", "32K 균일 길이: 분할 16 고정 regret 최댓값", 1.162, 0.0005, "x", _fixed("32k_uniform", "fixed:16", "loss_max"), FIXEDSPLIT, "1.162배"),
    Check("fixedsplit.32k.uniform.fixed1_max", "32K 균일 길이: 분할 1 고정 regret 최댓값", 4.861, 0.0005, "x", _fixed("32k_uniform", "fixed:1", "loss_max"), FIXEDSPLIT, "4.861배"),
    Check("fixedsplit.32k.union.fixed8_p90", "32K 혼합+균일 211셀: 분할 8 고정 regret 90퍼센타일", 1.034, 0.0005, "x", _fixed("32k_union", "fixed:8", "loss_p90"), FIXEDSPLIT, "1.034배"),
    Check("fixedsplit.32k.union.fixed8_max", "32K 혼합+균일 211셀: 분할 8 고정 regret 최댓값", 1.127, 0.0005, "x", _fixed("32k_union", "fixed:8", "loss_max"), FIXEDSPLIT, "1.127배"),
    Check("fixedsplit.32k.union.fixed8_over1_1", "32K 혼합+균일 211셀: 분할 8 고정 regret이 1.1배를 넘는 셀 수", 1, 0, "", _fixed("32k_union", "fixed:8", "over_1.1"), FIXEDSPLIT, "1셀"),
    Check("fixedsplit.32k.union.fixed16_max", "32K 혼합+균일 211셀: 분할 16 고정 regret 최댓값", 1.162, 0.0005, "x", _fixed("32k_union", "fixed:16", "loss_max"), FIXEDSPLIT, "1.162배"),
    Check("fixedsplit.64k.ragged.fixed8_max", "64K 혼합 길이 18셀: 분할 8 고정 regret 최댓값", 1.009, 0.0005, "x", _fixed("64k_ragged", "fixed:8", "loss_max"), FIXEDSPLIT, "1.009배"),
    Check("fixedsplit.64k.ragged.fixed16_max", "64K 혼합 길이: 분할 16 고정 regret 최댓값", 1.022, 0.0005, "x", _fixed("64k_ragged", "fixed:16", "loss_max"), FIXEDSPLIT, "1.022배"),
    Check("fixedsplit.32k.uniform_warm.fixed8_max", "32K 균일 길이 warm 49셀: 분할 8 고정 regret 최댓값", 1.712, 0.0005, "x", _fixed("32k_uniform_warm", "fixed:8", "loss_max"), FIXEDSPLIT, "1.712배"),
    Check("fixedsplit.32k.uniform_warm.fixed8_over1_25", "32K 균일 길이 warm: 분할 8 고정 regret이 1.25배를 넘는 셀 수", 8, 0, "", _fixed("32k_uniform_warm", "fixed:8", "over_1.25"), FIXEDSPLIT, "8셀"),
    Check("fixedsplit.s2.ragged.fixed8_max", "H_q=28·H_kv=4 격자 혼합 길이 135셀: 분할 8 고정 regret 최댓값", 1.146, 0.0005, "x", _fixed("s2_ragged", "fixed:8", "loss_max"), FIXEDSPLIT, "1.146배"),
    Check("fixedsplit.s2.ragged.fixed16_max", "H_q=28·H_kv=4 격자 혼합 길이: 분할 16 고정 regret 최댓값", 1.058, 0.0005, "x", _fixed("s2_ragged", "fixed:16", "loss_max"), FIXEDSPLIT, "1.058배"),
    Check("fixedsplit.s2.uniform.fixed8_max", "H_q=28·H_kv=4 격자 균일 길이 49셀: 분할 8 고정 regret 최댓값", 1.216, 0.0005, "x", _fixed("s2_uniform", "fixed:8", "loss_max"), FIXEDSPLIT, "1.216배"),
    Check("fixedsplit.32k.union.fa2_best_vs_best_max", "32K 혼합+균일 211셀: FA2 분할만 고르는 선택기 / 전체 최선(FlashInfer 포함) 최댓값", 1.106, 0.0005, "x", _fixed("32k_union", "fa2_best", "loss_max", "best"), FIXEDSPLIT, "1.106배"),
    Check("fixedsplit.32k.union.fixed8_vs_best_median", "32K 혼합+균일 211셀: 분할 8 고정 / 전체 최선(FlashInfer 포함) 중앙값", 1.037, 0.0005, "x", _fixed("32k_union", "fixed:8", "loss_median", "best"), FIXEDSPLIT, "1.037배"),

    Check("traceE2E.windows", "트레이스 창 선택: 후보 창 수(연속 300요청, 50건 간격)", 382, 0, "", _trace_selection("windows"), TRACE_E2E, "후보 창 382개 중 17개"),
    Check("traceE2E.windows_reaching_batch", "후보 창 중 캡 적용 replay 평균 배치가 26 이상인 창 수", 17, 0, "", _trace_selection("windows_reaching_batch"), TRACE_E2E, "후보 창 382개 중 17개"),
    Check("traceE2E.chosen_start", "선택한 창의 시작 요청 번호", 550, 0, "", _trace_selection("chosen_start"), TRACE_E2E, "요청 550..849"),
    Check("traceE2E.criterion_steps", "선택 기준: 예측 regret 1.25배 이상이면서 배치 26 이상인 step 수(시나리오에서 다시 계산)", 589, 0, "", _trace_e2e("criterion_steps"), TRACE_E2E, "기준 step 589개"),
    Check("traceE2E.requests", "트레이스 구동 시나리오의 요청 수", 300, 0, "", _trace_e2e("requests"), TRACE_E2E, "연속 300개 요청"),
    Check("traceE2E.steps", "시나리오 replay(논리 step 격자)의 decode step 수", 1897, 0, "", _trace_e2e("steps"), TRACE_E2E, "1,897 decode step"),
    Check("traceE2E.batch_mean", "시나리오 replay의 평균 배치", 26.9, 0.005, "", _trace_e2e("batch_mean"), TRACE_E2E, "평균 배치 26.9"),
    Check("traceE2E.attention_ratio", "모델 예측: 휴리스틱 대 최선 분할 attention 시간 비(시나리오 replay)", 1.187, 0.0005, "x", _trace_e2e("attention_time_ratio"), TRACE_E2E, "attention 시간 비 1.187"),
    Check("traceE2E.loss_ge_1_25_frac", "모델 예측: regret 1.25배 이상 step 비율", 0.3105, 0.0001, "", _trace_e2e("loss_ge_1.25_frac"), TRACE_E2E, "regret 1.25배 이상 step 31.0%"),
    Check("traceE2E.no_split_frac", "모델 예측: 휴리스틱이 분할 없이 도는 step 비율", 0.6795, 0.0001, "", _trace_e2e("no_split_frac"), TRACE_E2E, "분할 없이 도는 step 67.9%"),
    Check("traceE2E.prompt_max", "시나리오의 최대 프롬프트 토큰", 4119, 0, "", _trace_e2e("prompt_max"), TRACE_E2E, "최대 프롬프트 4,119 토큰"),
    Check("traceE2E.prompts_ge_4096", "시나리오에서 프롬프트가 4,096 토큰 이상인 요청 수", 4, 0, "", _trace_e2e("prompts_ge_4096"), TRACE_E2E, "4,096 이상은 4건뿐"),
    Check("traceE2E.output_capped", "출력 상한 256이 적용된 요청 수(번들의 예측 파일)", 129, 0, "", _trace_predicted("output_capped"), TRACE_E2E, "출력 상한 256 적용 129건"),
    Check("traceE2E.prompt_capped", "프롬프트 상한이 적용된 요청 수", 0, 0, "", _trace_predicted("prompt_capped"), TRACE_E2E, "상한 적용 0건"),
    Check("traceE2E.bundle_matches_recompute", "번들의 예측 파일 = 시나리오를 지금 replay한 결과(step 수·attention 시간 비, 1=같음)", 1, 0, "", _trace_e2e("bundle_matches_recompute"), TRACE_E2E, "attention 시간 비 1.187"),
    Check("traceE2E.not_relaxed", "창 선택이 배치 기준을 낮추지 않고 끝남(0=기준 충족)", 0, 0, "", _trace_selection("relaxed"), TRACE_E2E, "17개가 평균 배치 26 이상"),

    Check("vllmbackend.engine_starts", "10월 6일 vLLM 재현 로그 3개에서 백엔드를 보고한 엔진 시작 횟수", 8, 0, "", _vllm_backend("starts"), TRACE_E2E, "엔진 시작 8회 모두 FLASH_ATTN"),
    Check("vllmbackend.flash_attn_starts", "그중 FLASH_ATTN을 쓴 횟수", 8, 0, "", _vllm_backend("flash_attn"), TRACE_E2E, "엔진 시작 8회 모두 FLASH_ATTN"),
    Check("vllmbackend.flashinfer_starts", "그중 FLASHINFER를 쓴 횟수", 0, 0, "", _vllm_backend("flashinfer"), TRACE_E2E, "FLASH_ATTN으로 돌았다"),
    Check("vllmbackend.unknown_env_warnings", "Unknown vLLM environment variable 경고 횟수", 8, 0, "", _vllm_backend("warnings"), TRACE_E2E, "경고 8회"),
    Check("vllmbackend.requested_flashinfer", "FLASHINFER를 요청한 자식 실행 횟수", 4, 0, "", _vllm_backend("requested_flashinfer"), TRACE_E2E, "FLASHINFER를 요청한 4회"),

    Check("longtext.long_len", "64K 자연어 시나리오의 긴 요청 길이", 65536, 0, "", _scenario_stat(LONGTEXT_SCENARIO, "long_len"), TRACE_E2E, "65,536 토큰"),
    Check("longtext.short_count", "64K 자연어 시나리오의 짧은 요청 수", 31, 0, "", _scenario_stat(LONGTEXT_SCENARIO, "short_count"), TRACE_E2E, "짧은 요청 31개(512 토큰)"),
    Check("longtext.short_len", "64K 자연어 시나리오의 짧은 요청 길이", 512, 0, "", _scenario_stat(LONGTEXT_SCENARIO, "short_len"), TRACE_E2E, "짧은 요청 31개(512 토큰)"),
    Check("gsm8k.questions", "GSM8K 시나리오의 문항 수", 200, 0, "", _scenario_stat(GSM8K_SCENARIO, "questions"), TRACE_E2E, "앞 200문항"),
    Check("gsm8k.groups", "GSM8K 시나리오의 묶음 수", 7, 0, "", _scenario_stat(GSM8K_SCENARIO, "groups"), TRACE_E2E, "7묶음(31×6 + 14)"),
    Check("gsm8k.first_group_questions", "첫 묶음의 문항 수", 31, 0, "", _scenario_stat(GSM8K_SCENARIO, "first_group_questions"), TRACE_E2E, "7묶음(31×6 + 14)"),
    Check("gsm8k.spacing", "묶음 도착 간격(논리 step)", 270, 0, "", _scenario_stat(GSM8K_SCENARIO, "spacing"), TRACE_E2E, "논리 step 270 간격"),
    Check("gsm8k.max_new", "GSM8K 시나리오의 요청당 생성 토큰", 256, 0, "", _scenario_stat(GSM8K_SCENARIO, "max_new"), TRACE_E2E, "요청마다 생성 256토큰"),
    Check("sweep.conv_x0.25.loss_ge_1.25", "부하 스윕, Azure conv ×0.25: regret 1.25배 이상 step 비율", 0.004, 0.0005, "", _sweep("azure_conv_x0.25", "losses", "loss_ge_1.25_frac"), SWEEP, "0.4%"),
    Check("sweep.conv_x0.5.loss_ge_1.25", "부하 스윕, Azure conv ×0.5: regret 1.25배 이상 step 비율", 0.024, 0.0005, "", _sweep("azure_conv_x0.5", "losses", "loss_ge_1.25_frac"), SWEEP, "2.4%"),
    Check("sweep.conv_x0.75.loss_ge_1.25", "부하 스윕, Azure conv ×0.75: regret 1.25배 이상 step 비율", 0.116, 0.0005, "", _sweep("azure_conv_x0.75", "losses", "loss_ge_1.25_frac"), SWEEP, "11.6%"),
    Check("sweep.conv_x1.loss_ge_1.25", "부하 스윕, Azure conv ×1: regret 1.25배 이상 step 비율", 0.135, 0.0005, "", _sweep("azure_conv_x1", "losses", "loss_ge_1.25_frac"), SWEEP, "13.5%"),
    Check("sweep.conv_x1.25.loss_ge_1.25", "부하 스윕, Azure conv ×1.25: regret 1.25배 이상 step 비율", 0.059, 0.0005, "", _sweep("azure_conv_x1.25", "losses", "loss_ge_1.25_frac"), SWEEP, "5.9%"),
    Check("sweep.conv_x2.loss_ge_1.25", "부하 스윕, Azure conv ×2: regret 1.25배 이상 step 비율", 0.041, 0.0005, "", _sweep("azure_conv_x2", "losses", "loss_ge_1.25_frac"), SWEEP, "4.1%"),
    Check("sweep.conv_x4.loss_ge_1.25", "부하 스윕, Azure conv ×4: regret 1.25배 이상 step 비율", 0.042, 0.0005, "", _sweep("azure_conv_x4", "losses", "loss_ge_1.25_frac"), SWEEP, "4.2%"),
    Check("sweep.conv_x1.batch_mean", "부하 스윕, Azure conv ×1: 평균 배치", 34.77, 0.005, "", _sweep("azure_conv_x1", "losses", "batch_mean"), SWEEP, "| 34.8 |"),
    Check("sweep.conv_x0.75.batch_mean", "부하 스윕, Azure conv ×0.75: 평균 배치(분할 없음 임계 26 바로 위)", 26.1, 0.005, "", _sweep("azure_conv_x0.75", "losses", "batch_mean"), SWEEP, "| 26.1 |"),
    Check("sweep.conv_x4.batch_mean", "부하 스윕, Azure conv ×4: 평균 배치(KV 예산이 상한)", 47.17, 0.005, "", _sweep("azure_conv_x4", "losses", "batch_mean"), SWEEP, "| 47.2 |"),
    Check("sweep.conv_x1.no_split_frac", "부하 스윕, Azure conv ×1: 휴리스틱이 분할하지 않는 step 비율", 0.895, 0.0005, "", _sweep("azure_conv_x1", "losses", "no_split_frac"), SWEEP, "89.5%"),
    Check("sweep.conv_x0.5.no_split_frac", "부하 스윕, Azure conv ×0.5: 휴리스틱이 분할하지 않는 step 비율", 0.062, 0.0005, "", _sweep("azure_conv_x0.5", "losses", "no_split_frac"), SWEEP, "6.2%"),
    Check("sweep.conv_x1.heuristic_ratio", "부하 스윕, Azure conv ×1: attention 시간 비(휴리스틱/최선)", 1.119, 0.0005, "x", _sweep("azure_conv_x1", "losses", "attention_time_ratio"), SWEEP, "1.119"),
    Check("sweep.conv_x0.25.heuristic_ratio", "부하 스윕, Azure conv ×0.25: attention 시간 비(휴리스틱/최선)", 1.07, 0.0005, "x", _sweep("azure_conv_x0.25", "losses", "attention_time_ratio"), SWEEP, "1.070"),
    Check("sweep.code_x16.loss_ge_1.25", "부하 스윕, Azure code ×16: regret 1.25배 이상 step 비율", 0.141, 0.0005, "", _sweep("azure_code_x16", "losses", "loss_ge_1.25_frac"), SWEEP, "14.1%"),
    Check("sweep.code_x2.loss_ge_1.25", "부하 스윕, Azure code ×2: regret 1.25배 이상 step 비율", 0.041, 0.0005, "", _sweep("azure_code_x2", "losses", "loss_ge_1.25_frac"), SWEEP, "4.1%"),
    Check("sweep.code_x32.loss_ge_1.25", "부하 스윕, Azure code ×32: regret 1.25배 이상 step 비율(포화 고원)", 0.14, 0.0005, "", _sweep("azure_code_x32", "losses", "loss_ge_1.25_frac"), SWEEP, "14.0%"),
    Check("sweep.burstgpt.max_loss_ge_1.25", "부하 스윕, BurstGPT 7일 ×25~400: regret 1.25배 이상 step 비율의 최댓값", 0.003, 0.0005, "", _sweep_extreme("burstgpt_7d_x", max, "losses", "loss_ge_1.25_frac"), SWEEP, "0.3%"),
    Check("sweep.runs", "부하 스윕 실행 수(Azure conv 9 + code 7 + BurstGPT 5)", 21, 0, "", lambda repo, data: float(len(list((data / "traffic_sweep").glob("*/summary.json")))), SWEEP, "21개 실행"),
    Check("sweep.all.loss_ge_1.25_max", "부하 스윕 21개 실행에서 regret 1.25배 이상 step 비율의 최댓값", 0.141, 0.0005, "", _sweep_extreme("", max, "losses", "loss_ge_1.25_frac"), SWEEP, "code ×16 이후의 14.0~14.1%"),
    Check("sweep.repro.azure_conv_x0.5", "부하 스윕 재실행 azure_conv_x0.5: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_conv_x0.5"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.azure_conv_x1", "부하 스윕 재실행 azure_conv_x1: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_conv_x1"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.azure_conv_x2", "부하 스윕 재실행 azure_conv_x2: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_conv_x2"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.azure_code_x4", "부하 스윕 재실행 azure_code_x4: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_code_x4"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.azure_code_x8", "부하 스윕 재실행 azure_code_x8: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_code_x8"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.azure_code_x16", "부하 스윕 재실행 azure_code_x16: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("azure_code_x16"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.burstgpt_7d_x50", "부하 스윕 재실행 burstgpt_7d_x50: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("burstgpt_7d_x50"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.repro.burstgpt_7d_x100", "부하 스윕 재실행 burstgpt_7d_x100: 2026-10-06 번들과 손실 필드·step 수의 차이(0이어야 함)", 0, 1e-09, "", _sweep_repro("burstgpt_7d_x100"), SWEEP, "한 자리도 다르지 않음"),
    Check("sweep.fixed.azure_conv_x1.s8.ratio_best", "부하 스윕, azure_conv_x1: 분할 8 고정의 ratio_best", 1.034, 0.0005, "x", _sweep("azure_conv_x1", "fixed", "s8.ratio_best"), SWEEP, "1.034"),
    Check("sweep.fixed.azure_conv_x1.s8.ratio_heuristic", "부하 스윕, azure_conv_x1: 분할 8 고정의 ratio_heuristic", 0.924, 0.0005, "x", _sweep("azure_conv_x1", "fixed", "s8.ratio_heuristic"), SWEEP, "0.924"),
    Check("sweep.fixed.azure_conv_x1.s16.ratio_best", "부하 스윕, azure_conv_x1: 분할 16 고정의 ratio_best", 1.09, 0.0005, "x", _sweep("azure_conv_x1", "fixed", "s16.ratio_best"), SWEEP, "1.090"),
    Check("sweep.fixed.azure_conv_x0.25.s8.ratio_best", "부하 스윕, azure_conv_x0.25: 분할 8 고정의 ratio_best", 1.063, 0.0005, "x", _sweep("azure_conv_x0.25", "fixed", "s8.ratio_best"), SWEEP, "1.063"),
    Check("sweep.fixed.azure_conv_x0.25.s16.ratio_best", "부하 스윕, azure_conv_x0.25: 분할 16 고정의 ratio_best", 1.124, 0.0005, "x", _sweep("azure_conv_x0.25", "fixed", "s16.ratio_best"), SWEEP, "1.124"),
    Check("sweep.fixed.azure_conv_x0.25.s8.ratio_heuristic", "부하 스윕, azure_conv_x0.25: 분할 8 고정의 ratio_heuristic", 0.993, 0.0005, "x", _sweep("azure_conv_x0.25", "fixed", "s8.ratio_heuristic"), SWEEP, "0.993"),
    Check("sweep.fixed.azure_code_x16.s8.ratio_best", "부하 스윕, azure_code_x16: 분할 8 고정의 ratio_best", 1.005, 0.0005, "x", _sweep("azure_code_x16", "fixed", "s8.ratio_best"), SWEEP, "1.005"),
    Check("sweep.fixed.azure_code_x16.s8.ratio_heuristic", "부하 스윕, azure_code_x16: 분할 8 고정의 ratio_heuristic", 0.885, 0.0005, "x", _sweep("azure_code_x16", "fixed", "s8.ratio_heuristic"), SWEEP, "0.885"),
    Check("sweep.fixed.burstgpt_7d_x100.s8.ratio_best", "부하 스윕, burstgpt_7d_x100: 분할 8 고정의 ratio_best", 1.059, 0.0005, "x", _sweep("burstgpt_7d_x100", "fixed", "s8.ratio_best"), SWEEP, "1.059"),
    Check("sweep.fixed.burstgpt_7d_x100.s8.ratio_heuristic", "부하 스윕, burstgpt_7d_x100: 분할 8 고정의 ratio_heuristic", 1.039, 0.0005, "x", _sweep("burstgpt_7d_x100", "fixed", "s8.ratio_heuristic"), SWEEP, "1.039"),
    Check("sweep.fixed.burstgpt_7d_x400.s8.ratio_heuristic", "부하 스윕, burstgpt_7d_x400: 분할 8 고정의 ratio_heuristic", 1.057, 0.0005, "x", _sweep("burstgpt_7d_x400", "fixed", "s8.ratio_heuristic"), SWEEP, "1.057"),
    Check("sweep.fixed.azure_conv_x1.s8.ratio_best_b_lt_26", "부하 스윕, azure_conv_x1: 분할 8 고정의 ratio_best_b_lt_26", 1.023, 0.0005, "x", _sweep("azure_conv_x1", "fixed", "s8.ratio_best_b_lt_26"), SWEEP, "1.023"),
    Check("sweep.fixed.azure_conv_x1.s8.ratio_best_b_ge_26", "부하 스윕, azure_conv_x1: 분할 8 고정의 ratio_best_b_ge_26", 1.035, 0.0005, "x", _sweep("azure_conv_x1", "fixed", "s8.ratio_best_b_ge_26"), SWEEP, "1.035"),
    Check("sweep.fixed.s8.ratio_best_max", "부하 스윕 21개 실행에서 분할 8 고정 / 최선의 최댓값", 1.065, 0.0005, "x", _sweep_extreme("", max, "fixed", "s8.ratio_best"), SWEEP, "1.065"),
    Check("sweep.fixed.s8.ratio_heuristic_min", "부하 스윕 21개 실행에서 분할 8 고정 / 휴리스틱의 최솟값(가장 큰 이득)", 0.885, 0.0005, "x", _sweep_extreme("", min, "fixed", "s8.ratio_heuristic"), SWEEP, "0.885"),
    Check("sweep.fixed.s8.ratio_heuristic_max", "부하 스윕 21개 실행에서 분할 8 고정 / 휴리스틱의 최댓값(가장 큰 손해)", 1.057, 0.0005, "x", _sweep_extreme("", max, "fixed", "s8.ratio_heuristic"), SWEEP, "1.057"),
    Check("sweep.fixed.s8.regret_max_overall", "부하 스윕 21개 실행의 step 중 분할 8 고정의 단일 step 최악 regret", 1.36, 0.0005, "x", _sweep_extreme("", max, "fixed", "s8.regret_max"), SWEEP, "code ×8에서 1.360"),
    Check("sweep.azure_conv_x0.75.wait_p99_s", "부하 스윕, azure_conv_x0.75: 큐 대기 wait_p99_s", 0.03, 0.0005, "s", _sweep("azure_conv_x0.75", "load", "wait_p99_s"), SWEEP, "0.03"),
    Check("sweep.azure_conv_x1.wait_p99_s", "부하 스윕, azure_conv_x1: 큐 대기 wait_p99_s", 3.712, 0.0005, "s", _sweep("azure_conv_x1", "load", "wait_p99_s"), SWEEP, "p99 3.7 s"),
    Check("sweep.azure_conv_x1.25.wait_p99_s", "부하 스윕, azure_conv_x1.25: 큐 대기 wait_p99_s", 43.208, 0.0005, "s", _sweep("azure_conv_x1.25", "load", "wait_p99_s"), SWEEP, "| 43 |"),
    Check("sweep.azure_conv_x1.wait_gt_1s_frac", "부하 스윕, azure_conv_x1: 큐 대기 wait_gt_1s_frac", 0.055, 0.0005, "", _sweep("azure_conv_x1", "load", "wait_gt_1s_frac"), SWEEP, "대기 > 1 s 요청 5.5%"),
    Check("sweep.azure_conv_x0.75.wait_gt_1s_frac", "부하 스윕, azure_conv_x0.75: 큐 대기 wait_gt_1s_frac", 0.001, 0.0005, "", _sweep("azure_conv_x0.75", "load", "wait_gt_1s_frac"), SWEEP, "| 0.1% |"),
    Check("sweep.cond.conv_x0.5", "부하 스윕, Azure conv ×0.5: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.304, 0.0005, "", _sweep_cond("azure_conv_x0.5", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "30.4%"),
    Check("sweep.cond.conv_x0.75", "부하 스윕, Azure conv ×0.75: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.215, 0.0005, "", _sweep_cond("azure_conv_x0.75", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "21.5%"),
    Check("sweep.cond.conv_x1", "부하 스윕, Azure conv ×1: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.15, 0.0005, "", _sweep_cond("azure_conv_x1", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "15.0%"),
    Check("sweep.cond.conv_x1.25", "부하 스윕, Azure conv ×1.25: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.06, 0.0005, "", _sweep_cond("azure_conv_x1.25", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "6.0%"),
    Check("sweep.cond.conv_x2", "부하 스윕, Azure conv ×2: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.042, 0.0005, "", _sweep_cond("azure_conv_x2", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "4.2%"),
    Check("sweep.cond.code_x16", "부하 스윕, Azure code ×16: 배치 ≥ 26인 step 중 regret 1.25배 이상 비율", 0.2, 0.0005, "", _sweep_cond("azure_code_x16", "cond.loss_ge_1.25_given_b_ge_26"), SWEEP, "B ≥ 26인 step 중 code ×16은 20.0%"),
    Check("sweep.cond.conv_x1.b_ge_26", "부하 스윕, Azure conv ×1: 배치 ≥ 26인 step 비율", 0.895, 0.0005, "", _sweep_cond("azure_conv_x1", "cond.b_ge_26_frac"), SWEEP, "89.5%"),
    Check("sweep.bin.b13_25", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 13~25에서 regret 1.25배 이상 비율", 0.005, 0.0005, "", _sweep_bin(13, "loss_ge_1.25_frac"), SWEEP, "0.5%"),
    Check("sweep.bin.b26_29", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 26~29에서 regret 1.25배 이상 비율", 0.303, 0.0005, "", _sweep_bin(26, "loss_ge_1.25_frac"), SWEEP, "30.3%"),
    Check("sweep.bin.b30_34", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 30~34에서 regret 1.25배 이상 비율", 0.164, 0.0005, "", _sweep_bin(30, "loss_ge_1.25_frac"), SWEEP, "16.4%"),
    Check("sweep.bin.b35_39", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 35~39에서 regret 1.25배 이상 비율", 0.145, 0.0005, "", _sweep_bin(35, "loss_ge_1.25_frac"), SWEEP, "14.5%"),
    Check("sweep.bin.b40_44", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 40~44에서 regret 1.25배 이상 비율", 0.088, 0.0005, "", _sweep_bin(40, "loss_ge_1.25_frac"), SWEEP, "8.8%"),
    Check("sweep.bin.b45_49", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 45~49에서 regret 1.25배 이상 비율", 0.038, 0.0005, "", _sweep_bin(45, "loss_ge_1.25_frac"), SWEEP, "3.8%"),
    Check("sweep.bin.b50_64", "부하 스윕, Azure conv ×1·×2 합친 step: 배치 50~64에서 regret 1.25배 이상 비율", 0.012, 0.0005, "", _sweep_bin(50, "loss_ge_1.25_frac"), SWEEP, "1.2%"),
    # ---- P3: GPU queue results (2026-10-24/25) ------------------------------------------------------------------------
    # §1 fixed split in the engine (fixedsplit_20261024): TPOT ratios against the heuristic, mean of 3 repeats
    *[Check(f"p3fixed.{sc}.{pol}.speedup", f"고정 분할 실생성 {sc}: {pol}의 TPOT 배율(휴리스틱 TPOT / 정책 TPOT, 3회 평균)", exp, 0.0005, "x",
            _p3_serve(FIXED_RUN, sc, pol, "speedup_vs_heuristic"), FIXEDSPLIT, txt)
      for sc, pol, exp, txt in [
          ("ragged", "table", 1.8125, "table 1.813배"), ("ragged", "fixed8", 1.6445, "고정 8 1.644배"), ("ragged", "fixed16", 1.8039, "고정 16 1.804배"),
          ("uniform", "table", 1.0029, "table 1.003배"), ("uniform", "fixed8", 0.9980, "고정 8 0.998배"), ("uniform", "fixed16", 0.9895, "고정 16 0.990배"),
          ("arrivals", "table", 1.1728, "table 1.173배"), ("arrivals", "fixed8", 1.1783, "고정 8 1.178배"), ("arrivals", "fixed16", 1.1677, "고정 16 1.168배"),
          ("ragged_64k", "table", 2.4484, "table 2.448배"), ("ragged_64k", "fixed8", 2.4430, "고정 8 2.443배"), ("ragged_64k", "fixed16", 2.4286, "고정 16 2.429배")]],
    Check("p3fixed.ragged.fixed8.ci_low", "혼합 길이 고정 8의 반복 부트스트랩 구간 하한(이상치 반복 포함)", 1.3933, 0.0005, "x",
          _p3_serve(FIXED_RUN, "ragged", "fixed8", "speedup_ci95_low"), FIXEDSPLIT, "구간 1.393–1.815"),
    Check("p3fixed.ragged.fixed8.repeat1_tpot", "혼합 길이 고정 8의 반복 1 TPOT(ms; 반복 0·2는 34.1·34.0)", 45.173, 0.001, "ms",
          _p3_repeat(FIXED_RUN, "ragged", "fixed8", 1, "tpot_ms"), FIXEDSPLIT, "반복 1의 TPOT 45.17 ms"),
    Check("p3fixed.ragged.fixed8.repeat0_tpot", "혼합 길이 고정 8의 반복 0 TPOT(ms)", 34.115, 0.001, "ms",
          _p3_repeat(FIXED_RUN, "ragged", "fixed8", 0, "tpot_ms"), FIXEDSPLIT, "34.11 ms"),
    Check("p3fixed.ragged.fixed8.repeat2_tpot", "혼합 길이 고정 8의 반복 2 TPOT(ms)", 33.976, 0.001, "ms",
          _p3_repeat(FIXED_RUN, "ragged", "fixed8", 2, "tpot_ms"), FIXEDSPLIT, "33.98 ms"),
    Check("p3fixed.ragged.fixed8.pair02", "혼합 길이 고정 8: 이상치 반복 1을 뺀 반복 0·2의 TPOT 배율(휴리스틱 평균 / 고정 8 평균)", 1.8112, 0.0005, "x",
          lambda repo, data: float(_p3_repeats(data, FIXED_RUN, "ragged").loc["heuristic"].tpot_ms.iloc[[0, 2]].mean()
                                   / _p3_repeats(data, FIXED_RUN, "ragged").loc["fixed8"].tpot_ms.iloc[[0, 2]].mean()),
          FIXEDSPLIT, "반복 0·2만 보면 1.811배"),
    Check("p3fixed.ragged.heuristic.attn", "혼합 길이 휴리스틱의 attention ms/step", 36.806, 0.0005, "ms",
          _p3_serve(FIXED_RUN, "ragged", "heuristic", "attn_ms_per_step"), FIXEDSPLIT, "36.81 ms/step"),
    Check("p3fixed.ragged.fixed8.attn", "혼합 길이 고정 8의 attention ms/step(반복 1 포함)", 9.557, 0.0005, "ms",
          _p3_serve(FIXED_RUN, "ragged", "fixed8", "attn_ms_per_step"), FIXEDSPLIT, "9.56 ms/step"),
    Check("p3fixed.ragged.fixed8.repeat1_attn", "혼합 길이 고정 8 반복 1의 attention ms/step(다른 반복은 9.15)", 10.366, 0.0005, "ms",
          _p3_repeat(FIXED_RUN, "ragged", "fixed8", 1, "attn_ms"), FIXEDSPLIT, "반복 1의 attention 10.37 ms"),
    Check("p3fixed.ragged.table.attn", "혼합 길이 table의 attention ms/step", 9.247, 0.0005, "ms",
          _p3_serve(FIXED_RUN, "ragged", "table", "attn_ms_per_step"), FIXEDSPLIT, "9.25 ms/step"),
    Check("p3fixed.ragged_64k.heuristic.attn", "64K 혼합 길이 휴리스틱의 attention ms/step", 70.439, 0.0005, "ms",
          _p3_serve(FIXED_RUN, "ragged_64k", "heuristic", "attn_ms_per_step"), FIXEDSPLIT, "70.44 ms/step"),
    Check("p3fixed.ragged_64k.fixed8.attn", "64K 혼합 길이 고정 8의 attention ms/step", 14.342, 0.0005, "ms",
          _p3_serve(FIXED_RUN, "ragged_64k", "fixed8", "attn_ms_per_step"), FIXEDSPLIT, "14.34 ms/step"),
    Check("p3fixed.uniform.fixed8.events", "균일 길이 고정 8의 분기 지점 수(휴리스틱 대비)", 6, 0, "", _p3_event(FIXED_RUN, "uniform", "fixed8", "distinct_positions"), FIXEDSPLIT, "균일 길이에서 고정 8은 분기 지점 6곳"),
    Check("p3fixed.uniform.fixed8.tie", "그중 bf16 1 ulp 동점", 5, 0, "", _p3_event(FIXED_RUN, "uniform", "fixed8", "tie_1ulp"), FIXEDSPLIT, "동점(1 ulp) 5곳"),
    Check("p3fixed.uniform.fixed8.clear", "그중 clear", 1, 0, "", _p3_event(FIXED_RUN, "uniform", "fixed8", "clear"), FIXEDSPLIT, "clear 1곳"),
    Check("p3fixed.uniform.fixed16.events", "균일 길이 고정 16의 분기 지점 수", 6, 0, "", _p3_event(FIXED_RUN, "uniform", "fixed16", "distinct_positions"), FIXEDSPLIT, "고정 16도 같은 6곳"),
    Check("p3fixed.uniform.fixed8.mismatches", "균일 길이 고정 8의 불일치 토큰 수(반복당)", 114, 0, "", _p3_event(FIXED_RUN, "uniform", "fixed8", "token_mismatches_max"), FIXEDSPLIT, "불일치 토큰 114개"),
    Check("p3fixed.uniform.table.events", "균일 길이 table의 분기 지점 수", 0, 0, "", _p3_event(FIXED_RUN, "uniform", "table", "distinct_positions"), FIXEDSPLIT, "table은 균일 길이에서 0곳"),
    Check("p3fixed.ragged.events", "혼합 길이 table·고정 8·고정 16의 분기 지점 합", 0, 0, "",
          lambda repo, data: sum(_p3_event(FIXED_RUN, "ragged", p, "distinct_positions")(repo, data) for p in ("table", "fixed8", "fixed16")), FIXEDSPLIT, "혼합 길이에서는 세 정책 모두 분기 0곳"),
    Check("p3fixed.arrivals.events", "요청 도착 table·고정 8·고정 16의 분기 지점 합", 0, 0, "",
          lambda repo, data: sum(_p3_event(FIXED_RUN, "arrivals", p, "distinct_positions")(repo, data) for p in ("table", "fixed8", "fixed16")), FIXEDSPLIT, "요청 도착에서도 분기 0곳"),
    Check("p3fixed.ragged_64k.events", "64K 혼합 길이 table·고정 8·고정 16의 분기 지점 합(정책마다 1곳)", 3, 0, "",
          lambda repo, data: sum(_p3_event(FIXED_RUN, "ragged_64k", p, "distinct_positions")(repo, data) for p in ("table", "fixed8", "fixed16")), FIXEDSPLIT, "64K에서는 세 정책 모두 같은 한 곳"),
    Check("p3fixed.ragged_64k.clear", "64K 혼합 길이 table·고정 8·고정 16의 clear 합", 3, 0, "",
          lambda repo, data: sum(_p3_event(FIXED_RUN, "ragged_64k", p, "clear")(repo, data) for p in ("table", "fixed8", "fixed16")), FIXEDSPLIT, "요청 0 위치 27"),
    # §6 trace-replay scoring against measured kernel times (hw_4090/trace_validation_20261024)
    Check("p3val.conv.cells", "표본 셀 수(Azure conv ×1)", 100, 0, "", _p3_val("azure_conv_x1", "cells"), FIXEDSPLIT, "Azure conv ×1은 층마다 25개(100셀"),
    Check("p3val.conv.pred_ge", "conv ×1: 예측 regret ≥ 1.25인 표본 셀 수", 50, 0, "", _p3_val("azure_conv_x1", "pred_ge"), FIXEDSPLIT, "예측한 셀은 50개"),
    Check("p3val.conv.meas_ge", "conv ×1: 실측 regret ≥ 1.25인 표본 셀 수(1회차)", 41, 0, "", _p3_val("azure_conv_x1", "meas_ge"), FIXEDSPLIT, "그중 실측도 ≥ 1.25인 셀이 41개"),
    Check("p3val.conv.precision", "conv ×1 precision: 예측 ≥ 1.25 중 실측 ≥ 1.25 비율(표본)", 0.82, 0.0005, "", _p3_val("azure_conv_x1", "precision"), FIXEDSPLIT, "precision 0.82"),
    Check("p3val.conv.recall", "conv ×1 recall: 실측 ≥ 1.25 중 예측 ≥ 1.25 비율(표본)", 1.0, 0.0005, "", _p3_val("azure_conv_x1", "recall"), FIXEDSPLIT, "recall 1.00"),
    Check("p3val.conv.w_precision", "conv ×1: 층 가중 precision", 0.6565, 0.0005, "", _p3_val("azure_conv_x1", "w_precision"), FIXEDSPLIT, "층 가중 precision 0.657"),
    Check("p3val.conv.w_pred_frac", "conv ×1: 모델이 예측한 regret ≥ 1.25 step 비율(층 비율)", 0.1349, 0.0005, "", _p3_val("azure_conv_x1", "w_pred_frac"), FIXEDSPLIT, "예측 13.5%"),
    Check("p3val.conv.w_meas_frac", "conv ×1: 실측 regret ≥ 1.25 step 비율 추정(층 가중, 1회차)", 0.0886, 0.0005, "", _p3_val("azure_conv_x1", "w_meas_frac"), FIXEDSPLIT, "실측 기준 8.9%"),
    Check("p3val.conv.w_meas_frac_lo", "같은 추정의 95% 구간 하한(층 안 부트스트랩 2,000회)", 0.0628, 0.0005, "", _p3_val_ci("azure_conv_x1", "w_meas_frac", 0), FIXEDSPLIT, "95% 구간 6.3–10.9%"),
    Check("p3val.conv.w_meas_frac_hi", "같은 추정의 95% 구간 상한", 0.1091, 0.0005, "", _p3_val_ci("azure_conv_x1", "w_meas_frac", 1), FIXEDSPLIT, "95% 구간 6.3–10.9%"),
    Check("p3val.conv.w_meas_frac_pass2", "conv ×1: 2회차 측정의 실측 ≥ 1.25 step 비율 추정", 0.0937, 0.0005, "", _p3_val("azure_conv_x1", "w_meas_frac", 2), FIXEDSPLIT, "2회차(9.4%)"),
    Check("p3val.conv.w_meas_frac_keysteps", "conv ×1: 셀 가중을 같은 키의 step 수로 바꾼 추정", 0.1018, 0.0005, "", _p3_val("azure_conv_x1", "w_meas_frac", 1, "key_steps"), FIXEDSPLIT, "10.2%(구간 6.5–14.6%)"),
    Check("p3val.conv.att_meas", "conv ×1: 실측 attention 시간 비(휴리스틱 / 최선, 표본 가중)", 1.1439, 0.0005, "x", _p3_val("azure_conv_x1", "w_meas_attention_ratio"), FIXEDSPLIT, "실측 attention 시간 비는 1.144"),
    Check("p3val.conv.att_meas_lo", "같은 비의 95% 구간 하한", 1.1312, 0.0005, "x", _p3_val_ci("azure_conv_x1", "w_meas_attention_ratio", 0), FIXEDSPLIT, "구간 1.131–1.156"),
    Check("p3val.conv.att_meas_hi", "같은 비의 95% 구간 상한", 1.1564, 0.0005, "x", _p3_val_ci("azure_conv_x1", "w_meas_attention_ratio", 1), FIXEDSPLIT, "구간 1.131–1.156"),
    Check("p3val.conv.att_pred", "conv ×1: 같은 셀·가중의 예측 attention 시간 비", 1.1192, 0.0005, "x", _p3_val("azure_conv_x1", "w_pred_attention_ratio"), FIXEDSPLIT, "예측 1.119"),
    Check("p3val.conv.fp_min", "conv ×1 거짓 양성 9셀의 실측 regret 최솟값", 1.197, 0.0005, "x", _p3_val_false_positive_range("min"), FIXEDSPLIT, "실측 regret은 1.197–1.250"),
    Check("p3val.conv.fp_max", "conv ×1 거짓 양성 9셀의 실측 regret 최댓값", 1.2498, 0.0005, "x", _p3_val_false_positive_range("max"), FIXEDSPLIT, "실측 regret은 1.197–1.250"),
    Check("p3val.code.cells", "표본 셀 수(Azure code ×16)", 31, 0, "", _p3_val("azure_code_x16", "cells"), FIXEDSPLIT, "1셀, 31셀"),
    Check("p3val.code.recall", "code ×16 recall(표본)", 0.75, 0.0005, "", _p3_val("azure_code_x16", "recall"), FIXEDSPLIT, "recall 0.75"),
    Check("p3val.code.precision", "code ×16 precision(표본)", 0.8182, 0.0005, "", _p3_val("azure_code_x16", "precision"), FIXEDSPLIT, "precision 0.82(9/11)"),
    Check("p3val.code.w_meas_frac", "code ×16: 실측 ≥ 1.25 step 비율 추정(층 가중)", 0.2353, 0.0005, "", _p3_val("azure_code_x16", "w_meas_frac"), FIXEDSPLIT, "실측 23.5%"),
    Check("p3val.code.w_pred_frac", "code ×16: 예측 ≥ 1.25 step 비율", 0.1405, 0.0005, "", _p3_val("azure_code_x16", "w_pred_frac"), FIXEDSPLIT, "예측 14.1%"),
    Check("p3val.code.w_meas_frac_lo", "code ×16 실측 비율 구간 하한", 0.1254, 0.0005, "", _p3_val_ci("azure_code_x16", "w_meas_frac", 0), FIXEDSPLIT, "(12.5–37.2%)"),
    Check("p3val.code.w_meas_frac_hi", "code ×16 실측 비율 구간 상한", 0.3723, 0.0005, "", _p3_val_ci("azure_code_x16", "w_meas_frac", 1), FIXEDSPLIT, "(12.5–37.2%)"),
    Check("p3val.code.att_meas", "code ×16: 실측 attention 시간 비", 1.1606, 0.0005, "x", _p3_val("azure_code_x16", "w_meas_attention_ratio"), FIXEDSPLIT, "code ×16의 실측 attention 시간 비 1.161"),
    Check("p3val.err.abs_median", "전 표본: |예측 regret − 실측 regret| 중앙값", 0.0482, 0.0005, "", _p3_val_error("abs_err_median"), FIXEDSPLIT, "중앙값 0.048"),
    Check("p3val.err.abs_max", "전 표본: |예측 regret − 실측 regret| 최댓값", 0.2452, 0.0005, "", _p3_val_error("abs_err_max"), FIXEDSPLIT, "최댓값 0.245"),
    Check("p3val.err.rel_median", "전 표본: 예측 regret 상대 오차 중앙값", 0.0402, 0.0005, "", _p3_val_error("rel_err_median"), FIXEDSPLIT, "상대 오차 중앙값 4.0%"),
    Check("p3val.err.rel_max", "전 표본: 예측 regret 상대 오차 최댓값", 0.1693, 0.0005, "", _p3_val_error("rel_err_max"), FIXEDSPLIT, "최댓값 16.9%"),
    Check("p3val.err.spearman", "전 표본: 예측·실측 regret의 Spearman 순위 상관", 0.9585, 0.0005, "", _p3_val_error("spearman"), FIXEDSPLIT, "순위 상관 0.959"),
    Check("p3val.err.time_ratio", "전 표본: 휴리스틱 시간 예측 / 실측 중앙값", 0.8672, 0.0005, "", _p3_val_error("time_pred_over_meas_median"), FIXEDSPLIT, "절대 시간은 실측의 0.867배"),
    Check("p3val.err.best_variant", "전 표본: 예측 최선 분할 = 실측 최선 분할인 비율", 0.7099, 0.0005, "", _p3_val_error("best_variant_agreement"), FIXEDSPLIT, "같은 변형을 고른 비율 0.710"),
    Check("p3val.noise.regret_median", "두 측정 회차 사이 실측 regret 상대 차이 중앙값", 0.00155, 0.00005, "", _p3_val_noise("regret_rel_median"), FIXEDSPLIT, "중앙값 0.16%"),
    Check("p3val.noise.regret_max", "두 측정 회차 사이 실측 regret 상대 차이 최댓값", 0.00758, 0.00005, "", _p3_val_noise("regret_rel_max"), FIXEDSPLIT, "최댓값 0.76%"),
    *[Check(f"p3val.conv.stratum.{s}.{c}", f"conv ×1 층 {s}: {t}", exp, 0.0005, "", _p3_val_stratum("azure_conv_x1", s, c), FIXEDSPLIT, txt)
      for s, c, t, exp, txt in [
          ("<1.1", "meas_median", "실측 regret 중앙값", 1.0777, "층 <1.1의 실측 중앙값 1.078"),
          ("1.1-1.25", "meas_ge", "실측 ≥ 1.25 셀 수(25셀 중)", 0, "층 1.1–1.25는 0/25"),
          ("1.25-1.5", "meas_ge", "실측 ≥ 1.25 셀 수(25셀 중)", 16, "층 1.25–1.5는 16/25"),
          (">=1.5", "meas_ge", "실측 ≥ 1.25 셀 수(25셀 중)", 25, "층 ≥1.5는 25/25"),
          ("1.25-1.5", "pred_median", "예측 regret 중앙값", 1.3262, "예측 중앙값 1.326"),
          ("1.25-1.5", "meas_median", "실측 regret 중앙값", 1.2655, "실측 중앙값 1.265"),
          (">=1.5", "pred_median", "예측 regret 중앙값", 1.5739, "예측 중앙값 1.574"),
          (">=1.5", "meas_median", "실측 regret 중앙값", 1.4263, "실측 중앙값 1.426")]],
    # §7 trace-driven end to end (trace_e2e_20261024)
    *[Check(f"p3trace.{pol}.{what}", title, exp, tol, unit, fn, FIXEDSPLIT, txt)
      for pol in ("table",) for what, title, exp, tol, unit, fn, txt in [
          ("speedup", "table의 TPOT 배율(3회 평균)", 1.0346, 0.0005, "x", _p3_serve(TRACE_RUN, TRACE_NAME, "table", "speedup_vs_heuristic"), "table 1.035배"),
          ("attn", "table의 attention ms/step", 5.893, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "table", "attn_ms_per_step"), "table 5.89 ms"),
          ("p50", "table의 TPOT p50(ms)", 36.838, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "table", "tpot_ms_p50"), "p50 36.84"),
          ("p90", "table의 TPOT p90(ms)", 42.835, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "table", "tpot_ms_p90"), "p90 42.84"),
          ("throughput", "table의 처리량(토큰/s)", 797.8, 0.05, "tok/s", _p3_serve(TRACE_RUN, TRACE_NAME, "table", "throughput_tokens_s"), "797.8 tok/s"),
          ("step_ratio", "측정한 step별 attention 시간 합 비(휴리스틱 / table)", 1.1778, 0.0005, "x", _p3_step_ratio("table", "total"), "측정 비 1.178"),
          ("frac125", "측정한 step별 attention 배율이 1.25 이상인 step 비율(휴리스틱 / table)", 0.4308, 0.0005, "", _p3_step_ratio("table", "frac"), "43.1%")]],
    Check("p3trace.heuristic.attn", "휴리스틱의 attention ms/step", 6.941, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "heuristic", "attn_ms_per_step"), FIXEDSPLIT, "휴리스틱 6.94 ms"),
    Check("p3trace.heuristic.tpot", "휴리스틱의 TPOT(ms, 3회 평균)", 37.585, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "heuristic", "tpot_ms_mean"), FIXEDSPLIT, "휴리스틱 37.58 ms"),
    Check("p3trace.heuristic.p50", "휴리스틱의 TPOT p50(ms)", 38.443, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "heuristic", "tpot_ms_p50"), FIXEDSPLIT, "p50 38.44"),
    Check("p3trace.heuristic.p90", "휴리스틱의 TPOT p90(ms)", 44.5, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "heuristic", "tpot_ms_p90"), FIXEDSPLIT, "p90 44.50"),
    Check("p3trace.heuristic.throughput", "휴리스틱의 처리량(토큰/s)", 776.0, 0.05, "tok/s", _p3_serve(TRACE_RUN, TRACE_NAME, "heuristic", "throughput_tokens_s"), FIXEDSPLIT, "776.0에서 797.8"),
    Check("p3trace.model.attn", "model의 attention ms/step", 5.8, 0.0005, "ms", _p3_serve(TRACE_RUN, TRACE_NAME, "model", "attn_ms_per_step"), FIXEDSPLIT, "| model | 5.800 |"),
    Check("p3trace.model.step_ratio", "측정한 step별 attention 시간 합 비(휴리스틱 / model)", 1.1968, 0.0005, "x", _p3_step_ratio("model", "total"), FIXEDSPLIT, "측정 비 1.197"),
    Check("p3trace.model.frac125", "측정한 step별 배율이 1.25 이상인 step 비율(휴리스틱 / model)", 0.4613, 0.0005, "", _p3_step_ratio("model", "frac"), FIXEDSPLIT, "46.1%"),
    Check("p3trace.hybrid.step_ratio", "측정한 step별 attention 시간 합 비(휴리스틱 / hybrid)", 1.1811, 0.0005, "x", _p3_step_ratio("hybrid", "total"), FIXEDSPLIT, "측정 비 1.181"),
    Check("p3trace.model.r0_tpot", "model 반복 0의 TPOT(ms; 정책 캐시가 빈 첫 반복)", 43.615, 0.0005, "ms", _p3_repeat(TRACE_RUN, TRACE_NAME, "model", 0, "tpot_ms"), FIXEDSPLIT, "model 반복 0의 TPOT는 43.62 ms"),
    Check("p3trace.model.steady_tpot", "model 반복 1·2의 TPOT 평균(ms)", 36.252, 0.0005, "ms", _p3_steady_tpot(TRACE_RUN, TRACE_NAME, "model"), FIXEDSPLIT, "반복 1·2는 36.25 ms"),
    Check("p3trace.model.steady_speedup", "model 반복 1·2의 TPOT 배율(휴리스틱 반복 1·2 대비)", 1.0368, 0.0005, "x", _p3_steady_speedup(TRACE_RUN, TRACE_NAME, "model"), FIXEDSPLIT, "model 1.037배"),
    Check("p3trace.hybrid.r0_tpot", "hybrid 반복 0의 TPOT(ms)", 43.685, 0.0005, "ms", _p3_repeat(TRACE_RUN, TRACE_NAME, "hybrid", 0, "tpot_ms"), FIXEDSPLIT, "hybrid 43.69 ms"),
    Check("p3trace.hybrid.steady_speedup", "hybrid 반복 1·2의 TPOT 배율", 1.0347, 0.0005, "x", _p3_steady_speedup(TRACE_RUN, TRACE_NAME, "hybrid"), FIXEDSPLIT, "hybrid 1.035배"),
    Check("p3trace.model.all_speedup", "model의 3회 평균 TPOT 배율(반복 0 포함)", 0.971, 0.0005, "x", _p3_serve(TRACE_RUN, TRACE_NAME, "model", "speedup_vs_heuristic"), FIXEDSPLIT, "0.971배"),
    Check("p3trace.model.r0_policy_us", "model 반복 0의 step당 정책 비용(µs)", 6460.7, 0.5, "us", _p3_repeat(TRACE_RUN, TRACE_NAME, "model", 0, "policy_us"), FIXEDSPLIT, "6,461 µs"),
    Check("p3trace.model.r1_policy_us", "model 반복 1의 step당 정책 비용(µs)", 6.3, 0.05, "us", _p3_repeat(TRACE_RUN, TRACE_NAME, "model", 1, "policy_us"), FIXEDSPLIT, "6.3 µs"),
    Check("p3trace.model.r0_hit", "model 반복 0의 결정 캐시 적중률", 0.6568, 0.0005, "", _p3_repeat(TRACE_RUN, TRACE_NAME, "model", 0, "hit"), FIXEDSPLIT, "결정 캐시 적중률은 65.7%"),
    Check("p3trace.table.r0_policy_us", "table 반복 0의 step당 정책 비용(µs)", 29.2, 0.05, "us", _p3_repeat(TRACE_RUN, TRACE_NAME, "table", 0, "policy_us"), FIXEDSPLIT, "table은 29 µs"),
    Check("p3trace.table.events", "table의 분기 지점 수(휴리스틱 대비, 300요청 중 10개 요청)", 10, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "table", "distinct_positions"), FIXEDSPLIT, "table 분기 지점 10곳"),
    Check("p3trace.table.tie", "그중 교사 강제 구간 안이라 분류된 동점(1 ulp)", 1, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "table", "tie_1ulp"), FIXEDSPLIT, "분류된 것은 1곳"),
    Check("p3trace.table.unclassified", "그중 교사 강제 구간 밖이라 미분류", 9, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "table", "unclassified"), FIXEDSPLIT, "나머지 9곳"),
    Check("p3trace.table.mismatches", "table의 불일치 토큰 수(반복당)", 921, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "table", "token_mismatches_max"), FIXEDSPLIT, "불일치 토큰 table 921개"),
    Check("p3trace.model.events", "model의 분기 지점 수", 8, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "model", "distinct_positions"), FIXEDSPLIT, "model 8곳"),
    Check("p3trace.model.clear", "model의 clear 수", 0, 0, "", _p3_event(TRACE_RUN, TRACE_NAME, "model", "clear"), FIXEDSPLIT, "clear는 없다"),
    Check("p3trace.teacher_rows", "교사 강제로 기록된 정책당 (step, 요청) 행 수", 8684, 0, "", _p3_teacher(TRACE_RUN, TRACE_NAME, "rows"), FIXEDSPLIT, "8,684행"),
    Check("p3trace.teacher_flips", "그 행에서 argmax가 휴리스틱과 다른 수(table)", 1, 0, "", _p3_teacher(TRACE_RUN, TRACE_NAME, "flips"), FIXEDSPLIT, "argmax가 바뀐 행은 1개"),
    # §8 vLLM decode step decomposition (vllm_4090/profile_20261024)
    *[Check(f"p3vllm.{run}.{col}", f"vLLM 0.31 {run}: {t}", exp, tol, "ms", _p3_vllm_value(run, col), FIXEDSPLIT, txt)
      for run, col, t, exp, tol, txt in [
          ("flash_attn_eager_ragged", "unprofiled_ms", "비프로파일 step 시간", 81.013, 0.0005, "혼합 길이 step은 81.0 ms"),
          ("flash_attn_eager_ragged", "attention_ms", "attention GPU 시간", 67.73, 0.005, "attention 67.7 ms"),
          ("flash_attn_eager_ragged", "gemm_ms", "GEMM GPU 시간", 9.56, 0.005, "GEMM 9.6 ms"),
          ("flash_attn_eager_ragged", "gpu_busy_ms", "GPU busy", 78.19, 0.005, "| FA2 eager, 혼합 | 81.0 | 78.2 |"),
          ("flash_attn_eager_ragged", "not_gpu_busy_ms", "비프로파일 step 시간 − GPU busy", 2.82, 0.005, "GPU 밖 시간은 FA2 혼합 2.8 ms"),
          ("flash_attn_eager_ragged", "attention_share_of_busy", "attention이 GPU busy에서 차지하는 비중", 0.866, 0.0005, "GPU busy의 87%"),
          ("flash_attn_eager_uniform", "unprofiled_ms", "비프로파일 step 시간", 14.629, 0.0005, "FA2 균일 14.6 ms"),
          ("flash_attn_eager_uniform", "attention_ms", "attention GPU 시간", 2.95, 0.005, "균일은 2.95 ms다"),
          ("flash_attn_eager_uniform", "gemm_ms", "GEMM GPU 시간", 9.65, 0.005, "(GEMM 9.6 ms 대 9.7 ms)"),
          ("flash_attn_cudagraph_ragged", "unprofiled_ms", "비프로파일 step 시간", 80.574, 0.0005, "CUDA graph 혼합 80.6 ms"),
          ("flash_attn_cudagraph_ragged", "attention_ms", "attention GPU 시간", 67.72, 0.005, "attention 67.7 ms"),
          ("flash_attn_cudagraph_uniform", "unprofiled_ms", "비프로파일 step 시간", 13.69, 0.005, "CUDA graph 균일 13.7 ms"),
          ("flashinfer_eager_ragged", "unprofiled_ms", "비프로파일 step 시간", 35.955, 0.0005, "혼합 step 36.0 ms"),
          ("flashinfer_eager_ragged", "attention_ms", "attention GPU 시간", 24.03, 0.005, "attention이 24.0 ms"),
          ("flashinfer_eager_ragged", "attention_share_of_busy", "attention이 GPU busy에서 차지하는 비중", 0.702, 0.0005, "GPU busy의 70%"),
          ("flashinfer_eager_uniform", "unprofiled_ms", "비프로파일 step 시간", 14.33, 0.005, "균일 14.3 ms의 2.5배"),
          ("flashinfer_eager_uniform", "attention_ms", "attention GPU 시간", 2.96, 0.005, "| FlashInfer eager, 균일 | 14.3 | 13.3 | 2.96 |")]],
    Check("p3vllm.excess_share.fa2", "FA2 eager: 혼합−균일 step 시간 차 중 attention GPU 시간 차의 비율", 0.9758, 0.0005, "",
          _p3_vllm_excess_share("flash_attn_eager_ragged", "flash_attn_eager_uniform"), FIXEDSPLIT, "차이의 97.6%"),
    Check("p3vllm.excess_share.fi", "FlashInfer eager: 같은 비율", 0.9742, 0.0005, "",
          _p3_vllm_excess_share("flashinfer_eager_ragged", "flashinfer_eager_uniform"), FIXEDSPLIT, "차이의 97.4%"),
    Check("p3vllm.after_leave.attention", "32K 요청이 끝난 직후 decode iteration(31개 요청)의 attention ms", 2.89, 0.005, "ms",
          _p3_vllm_after_leave("attention_us"), FIXEDSPLIT, "2.89 ms로 떨어진다"),
    Check("p3vllm.logged_median.attention", "analyze가 기록한 중앙값 칸의 FA2 혼합 attention ms(iteration 종류가 섞인 값, 사용하지 않음)", 1.44, 0.005, "ms",
          _p3_vllm_logged_median("flash_attn_eager_ragged", "attention_us_median"), FIXEDSPLIT, "기록된 중앙값 1.44 ms"),
    Check("p3vllm.grid_ratio", "vLLM FA2 attention(층당) / 같은 배치의 커널 격자 휴리스틱 시간", 1.7554, 0.0005, "x", _p3_vllm_grid_ratio, FIXEDSPLIT, "격자의 1.76배"),
    Check("p3vllm.grid_best", "같은 배치의 격자에서 휴리스틱 / 최선 FA2 분할 시간", 3.8, 0.05, "x", _p3_vllm_best_over_grid, FIXEDSPLIT, "격자의 최선 분할은 휴리스틱의 1/3.8"),
    # §9 64K natural text (longctx_text_20261024)
    Check("p3text.table.speedup", "64K 자연어: table TPOT 배율", 2.449, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "table", "speedup_vs_heuristic"), FIXEDSPLIT, "table 2.449배"),
    Check("p3text.model.speedup", "64K 자연어: model TPOT 배율", 2.4433, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "model", "speedup_vs_heuristic"), FIXEDSPLIT, "model 2.443배"),
    Check("p3text.hybrid.speedup", "64K 자연어: hybrid TPOT 배율", 2.4402, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "hybrid", "speedup_vs_heuristic"), FIXEDSPLIT, "hybrid 2.440배"),
    Check("p3text.heuristic.attn", "64K 자연어: 휴리스틱 attention ms/step", 70.453, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "heuristic", "attn_ms_per_step"), FIXEDSPLIT, "70.45 ms/step"),
    Check("p3text.table.attn", "64K 자연어: table attention ms/step", 14.335, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "table", "attn_ms_per_step"), FIXEDSPLIT, "14.33 ms/step"),
    Check("p3text.events", "64K 자연어: 세 정책의 분기 지점 합", 0, 0, "",
          lambda repo, data: sum(_p3_event(LONGTEXT_RUN, "demo_text_ragged_64k", p, "distinct_positions")(repo, data) for p in ("table", "model", "hybrid")), FIXEDSPLIT, "분기 지점 0곳"),
    Check("p3text.mismatches", "64K 자연어: 세 정책의 불일치 토큰 합", 0, 0, "",
          lambda repo, data: sum(_p3_event(LONGTEXT_RUN, "demo_text_ragged_64k", p, "token_mismatches_max")(repo, data) for p in ("table", "model", "hybrid")), FIXEDSPLIT, "불일치 토큰 0개"),
    Check("p3text.teacher_rows", "64K 자연어: 교사 강제 행 수(정책당)", 2016, 0, "", _p3_teacher(LONGTEXT_RUN, "demo_text_ragged_64k", "rows"), FIXEDSPLIT, "2,016행"),
    Check("p3text.teacher_flips", "64K 자연어: argmax가 달라진 행 수", 0, 0, "", _p3_teacher(LONGTEXT_RUN, "demo_text_ragged_64k", "flips"), FIXEDSPLIT, "argmax 변화 0"),
    Check("p3text.teacher_maxdiff", "64K 자연어: 교사 강제 로짓 최대 차이", 0.5, 0.005, "", _p3_teacher(LONGTEXT_RUN, "demo_text_ragged_64k", "max_diff"), FIXEDSPLIT, "로짓 최대 차이 0.5"),
    Check("p3text.teacher_tied", "64K 자연어: 휴리스틱 기준 1위 동점인 행 수(table)", 15, 0, "", _p3_teacher(LONGTEXT_RUN, "demo_text_ragged_64k", "tied"), FIXEDSPLIT, "1위 동점 15행"),
    # §10 GSM8K (gsm8k_20261024)
    *[Check(f"p3gsm.{pol}.speedup", f"GSM8K 시나리오: {pol}의 TPOT 배율", exp, 0.0005, "x", _p3_serve(GSM8K_RUN, "gsm8k_ragged32k_n200", pol, "speedup_vs_heuristic"), FIXEDSPLIT, txt)
      for pol, exp, txt in [("table", 2.1009, "table 2.101배"), ("fixed8", 2.1541, "고정 8 2.154배"), ("flashinfer_cudacore", 2.1553, "flashinfer_cudacore 2.155배")]],
    Check("p3gsm.heuristic.attn", "GSM8K 시나리오: 휴리스틱 attention ms/step", 32.905, 0.0005, "ms", _p3_serve(GSM8K_RUN, "gsm8k_ragged32k_n200", "heuristic", "attn_ms_per_step"), FIXEDSPLIT, "32.91 ms/step"),
    Check("p3gsm.acc.marker", "GSM8K 200문항: `####` 뒤 숫자 기준 정확도(네 정책 모두)", 0.78, 0.0005, "", _p3_acc("heuristic", "marker", "accuracy"), FIXEDSPLIT, "정확도 0.780"),
    Check("p3gsm.acc.marker_lo", "같은 정확도의 95% Wilson 구간 하한", 0.7176, 0.0005, "", _p3_acc("heuristic", "marker", "ci_low"), FIXEDSPLIT, "0.718–0.832"),
    Check("p3gsm.acc.marker_hi", "같은 정확도의 95% Wilson 구간 상한", 0.8318, 0.0005, "", _p3_acc("heuristic", "marker", "ci_high"), FIXEDSPLIT, "0.718–0.832"),
    *[Check(f"p3gsm.acc.marker.{pol}", f"GSM8K: {pol}의 `####` 기준 정답 수", 156, 0, "", _p3_acc(pol, "marker", "correct"), FIXEDSPLIT, "네 정책 모두 156문항")
      for pol in ("table", "fixed8", "flashinfer_cudacore")],
    Check("p3gsm.acc.last.heuristic", "GSM8K: 마지막 숫자 기준 휴리스틱 정확도", 0.395, 0.0005, "", _p3_acc("heuristic", "last", "accuracy"), FIXEDSPLIT, "마지막 숫자 기준 0.395"),
    Check("p3gsm.acc.last.cudacore", "GSM8K: 마지막 숫자 기준 flashinfer_cudacore 정확도", 0.405, 0.0005, "", _p3_acc("flashinfer_cudacore", "last", "accuracy"), FIXEDSPLIT, "flashinfer_cudacore 0.405"),
    Check("p3gsm.pair.cudacore.r2w", "flashinfer_cudacore: `####` 기준 맞음→틀림 문항 수", 3, 0, "", _p3_paired("flashinfer_cudacore", "marker", "right_to_wrong"), FIXEDSPLIT, "맞음→틀림 3, 틀림→맞음 3"),
    Check("p3gsm.pair.cudacore.w2r", "flashinfer_cudacore: `####` 기준 틀림→맞음 문항 수", 3, 0, "", _p3_paired("flashinfer_cudacore", "marker", "wrong_to_right"), FIXEDSPLIT, "맞음→틀림 3, 틀림→맞음 3"),
    Check("p3gsm.pair.cudacore.changed", "flashinfer_cudacore: `####` 답이 휴리스틱과 다른 문항 수", 8, 0, "", _p3_paired("flashinfer_cudacore", "marker", "prediction_changed"), FIXEDSPLIT, "답이 달라진 문항 8"),
    Check("p3gsm.pair.cudacore.text", "flashinfer_cudacore: 생성 토큰이 휴리스틱과 다른 문항 수(200문항 중)", 140, 0, "", _p3_paired("flashinfer_cudacore", "marker", "text_differs"), FIXEDSPLIT, "생성이 달라진 문항 140"),
    Check("p3gsm.pair.cudacore.same_answer", "그중 `####` 답은 같은 문항 수", 132, 0, "", _p3_paired("flashinfer_cudacore", "marker", "text_differs_answer_same"), FIXEDSPLIT, "그중 132문항은 답이 같다"),
    Check("p3gsm.pair.cudacore.last_r2w", "flashinfer_cudacore: 마지막 숫자 기준 맞음→틀림", 13, 0, "", _p3_paired("flashinfer_cudacore", "last", "right_to_wrong"), FIXEDSPLIT, "13 / 15"),
    Check("p3gsm.pair.cudacore.last_w2r", "flashinfer_cudacore: 마지막 숫자 기준 틀림→맞음", 15, 0, "", _p3_paired("flashinfer_cudacore", "last", "wrong_to_right"), FIXEDSPLIT, "13 / 15"),
    Check("p3gsm.pair.cudacore.last_p", "flashinfer_cudacore: 마지막 숫자 기준 정확 McNemar p", 0.8506, 0.0005, "", _p3_paired("flashinfer_cudacore", "last", "p_value"), FIXEDSPLIT, "p = 0.851"),
    Check("p3gsm.pair.table.changed", "table: 답이 달라진 문항 수", 0, 0, "", _p3_paired("table", "marker", "prediction_changed"), FIXEDSPLIT, "table·고정 8은 200문항 모두 생성 토큰이 같다"),
    Check("p3gsm.pair.table.text", "table: 생성 토큰이 다른 문항 수", 0, 0, "", _p3_paired("table", "marker", "text_differs"), FIXEDSPLIT, "table·고정 8은 200문항 모두 생성 토큰이 같다"),
    Check("p3gsm.pair.fixed8.text", "고정 8: 생성 토큰이 다른 문항 수", 0, 0, "", _p3_paired("fixed8", "marker", "text_differs"), FIXEDSPLIT, "table·고정 8은 200문항 모두 생성 토큰이 같다"),
    Check("p3gsm.events.cudacore.answered", "flashinfer_cudacore: 채점 대상 200문항에서의 분기 지점 수(반복 0)", 140, 0, "", _p3_gsm8k_events("flashinfer_cudacore", True), FIXEDSPLIT, "채점 문항 140곳"),
    Check("p3gsm.events.cudacore.filler", "flashinfer_cudacore: 32K 필러 요청 7개에서의 분기 지점 수", 7, 0, "", _p3_gsm8k_events("flashinfer_cudacore", False), FIXEDSPLIT, "필러 7곳"),
    Check("p3gsm.events.table.answered", "table: 채점 문항의 분기 지점 수", 0, 0, "", _p3_gsm8k_events("table", True), FIXEDSPLIT, "채점 문항의 분기는 0곳"),
    Check("p3gsm.events.table.filler", "table: 필러 요청의 분기 지점 수(7개 중)", 6, 0, "", _p3_gsm8k_events("table", False), FIXEDSPLIT, "필러 7개 중 6개"),
    Check("p3gsm.events.fixed8.filler", "고정 8: 필러 요청의 분기 지점 수", 6, 0, "", _p3_gsm8k_events("fixed8", False), FIXEDSPLIT, "필러 7개 중 6개"),
    # ---- P3 additions: every other number of the §1/§6-§10 tables ---------------------------------------------------
    *[Check(f"p3fixed.ci.{sc}.{pol}.{side}", f"고정 분할 실생성 {sc}: {pol} TPOT 배율 구간 {side}", exp, 0.0006, "x", _p3_serve(FIXED_RUN, sc, pol, f"speedup_ci95_{side}"), FIXEDSPLIT, txt)
      for sc, pol, side, exp, txt in [
          ("ragged", "table", "low", 1.7952, "1.813 (1.795–1.830)"), ("ragged", "table", "high", 1.8299, "1.813 (1.795–1.830)"),
          ("ragged", "fixed16", "low", 1.7902, "1.804 (1.790–1.827)"), ("ragged", "fixed16", "high", 1.8273, "1.804 (1.790–1.827)"),
          ("ragged", "fixed8", "high", 1.8147, "1.644 (1.393–1.815)"),
          ("uniform", "table", "low", 0.9990, "1.003 (0.999–1.008)"), ("uniform", "table", "high", 1.0080, "1.003 (0.999–1.008)"),
          ("uniform", "fixed8", "low", 0.9957, "0.998 (0.996–1.002)"), ("uniform", "fixed8", "high", 1.0015, "0.998 (0.996–1.002)"),
          ("uniform", "fixed16", "low", 0.9869, "0.990 (0.987–0.994)"), ("uniform", "fixed16", "high", 0.9939, "0.990 (0.987–0.994)"),
          ("arrivals", "table", "low", 1.1697, "1.173 (1.170–1.177)"), ("arrivals", "table", "high", 1.1769, "1.173 (1.170–1.177)"),
          ("arrivals", "fixed8", "low", 1.1755, "1.178 (1.176–1.180)"), ("arrivals", "fixed8", "high", 1.1798, "1.178 (1.176–1.180)"),
          ("arrivals", "fixed16", "low", 1.1662, "1.168 (1.166–1.169)"), ("arrivals", "fixed16", "high", 1.1692, "1.168 (1.166–1.169)"),
          ("ragged_64k", "table", "low", 2.4456, "2.448 (2.446–2.451)"), ("ragged_64k", "table", "high", 2.4510, "2.448 (2.446–2.451)"),
          ("ragged_64k", "fixed8", "low", 2.4424, "2.443 (2.442–2.444)"), ("ragged_64k", "fixed8", "high", 2.4441, "2.443 (2.442–2.444)"),
          ("ragged_64k", "fixed16", "low", 2.4237, "2.429 (2.424–2.431)"), ("ragged_64k", "fixed16", "high", 2.4311, "2.429 (2.424–2.431)")]],
    Check("p3fixed.ragged.heuristic.repeat1_tpot", "혼합 길이 휴리스틱의 반복 1 TPOT(ms)", 62.9375, 0.001, "ms", _p3_repeat(FIXED_RUN, "ragged", "heuristic", 1, "tpot_ms"), FIXEDSPLIT, "62.94 ms"),
    Check("p3fixed.ragged.fixed8.repeat1_other", "혼합 길이 고정 8 반복 1의 step 시간 − attention(ms)", 24.191, 0.005, "ms",
          lambda repo, data: float(_p3_repeats(data, FIXED_RUN, "ragged").loc[("fixed8", 1), "step_ms"] - _p3_repeats(data, FIXED_RUN, "ragged").loc[("fixed8", 1), "attn_ms"]),
          FIXEDSPLIT, "step 시간 − attention 24.2 ms"),
    Check("p3fixed.ragged.fixed8.repeat0_other", "혼합 길이 고정 8 반복 0의 step 시간 − attention(ms)", 14.771, 0.005, "ms",
          lambda repo, data: float(_p3_repeats(data, FIXED_RUN, "ragged").loc[("fixed8", 0), "step_ms"] - _p3_repeats(data, FIXED_RUN, "ragged").loc[("fixed8", 0), "attn_ms"]),
          FIXEDSPLIT, "다른 반복 14.8 ms"),
    Check("p3fixed.ragged.fixed8.repeat0_attn", "혼합 길이 고정 8 반복 0의 attention ms/step", 9.151, 0.0005, "ms", _p3_repeat(FIXED_RUN, "ragged", "fixed8", 0, "attn_ms"), FIXEDSPLIT, "다른 반복 9.15 ms"),
    Check("p3fixed.uniform.table.splits", "균일 길이 table이 고른 분할 수의 서로 다른 값 개수(첫 반복, 1 = 모든 step이 같은 값)", 1, 0, "",
          lambda repo, data: float(pd.read_parquet(data / "serve_4090" / FIXED_RUN / "uniform" / "table" / "repeat_000" / "steps.parquet").num_splits.nunique()), FIXEDSPLIT, "63 step 모두 휴리스틱(`num_splits=0`)"),
    Check("p3fixed.uniform.table.splits_value", "균일 길이 table이 고른 분할 수(0 = 휴리스틱)", 0, 0, "",
          lambda repo, data: float(pd.read_parquet(data / "serve_4090" / FIXED_RUN / "uniform" / "table" / "repeat_000" / "steps.parquet").num_splits.iloc[0]), FIXEDSPLIT, "63 step 모두 휴리스틱(`num_splits=0`)"),
    Check("p3fixed.ragged.table.splits_value", "혼합 길이 table이 고른 분할 수(모든 step 같은 값)", 16, 0, "",
          lambda repo, data: float(pd.read_parquet(data / "serve_4090" / FIXED_RUN / "ragged" / "table" / "repeat_000" / "steps.parquet").num_splits.iloc[0]
                                   if pd.read_parquet(data / "serve_4090" / FIXED_RUN / "ragged" / "table" / "repeat_000" / "steps.parquet").num_splits.nunique() == 1 else -1),
          FIXEDSPLIT, "혼합 길이는 63 step 모두 16"),
    Check("p3fixed.arrivals.table.splits8", "요청 도착 table이 분할 8을 고른 step 수(63 중)", 41, 0, "",
          lambda repo, data: float((pd.read_parquet(data / "serve_4090" / FIXED_RUN / "arrivals" / "table" / "repeat_000" / "steps.parquet").num_splits == 8).sum()), FIXEDSPLIT, "8이 41 step"),
    Check("p3fixed.ragged_64k.table.splits_value", "64K 혼합 길이 table이 고른 분할 수(모든 step 같은 값)", 8, 0, "",
          lambda repo, data: float(pd.read_parquet(data / "serve_4090" / FIXED_RUN / "ragged_64k" / "table" / "repeat_000" / "steps.parquet").num_splits.iloc[0]
                                   if pd.read_parquet(data / "serve_4090" / FIXED_RUN / "ragged_64k" / "table" / "repeat_000" / "steps.parquet").num_splits.nunique() == 1 else -1),
          FIXEDSPLIT, "64K 혼합 길이는 63 step 모두 8"),
    Check("p3val.conv.att_meas_pass2", "conv ×1: 2회차 실측 attention 시간 비", 1.1434, 0.0005, "x", _p3_val("azure_conv_x1", "w_meas_attention_ratio", 2), FIXEDSPLIT, "1.143"),
    Check("p3val.conv.att_meas_keysteps", "conv ×1: 키 step 수 가중 실측 attention 시간 비", 1.1391, 0.0005, "x", _p3_val("azure_conv_x1", "w_meas_attention_ratio", 1, "key_steps"), FIXEDSPLIT, "1.139"),
    Check("p3val.conv.w_meas_frac_keysteps_lo", "같은 추정의 구간 하한(키 step 수 가중)", 0.0649, 0.0006, "",
          lambda repo, data: float(__import__("kernelscope.analysis.validation", fromlist=["x"]).bootstrap(_p3_validation(data, "key_steps")[0], "azure_conv_x1", "w_meas_frac")[0]), FIXEDSPLIT, "10.2%(구간 6.5–14.6%)"),
    Check("p3val.conv.w_meas_frac_keysteps_hi", "같은 추정의 구간 상한(키 step 수 가중)", 0.1457, 0.0006, "",
          lambda repo, data: float(__import__("kernelscope.analysis.validation", fromlist=["x"]).bootstrap(_p3_validation(data, "key_steps")[0], "azure_conv_x1", "w_meas_frac")[1]), FIXEDSPLIT, "10.2%(구간 6.5–14.6%)"),
    Check("p3val.conv.w_precision_lo", "conv ×1 층 가중 precision 구간 하한", 0.4657, 0.0006, "", _p3_val_ci("azure_conv_x1", "w_precision", 0), FIXEDSPLIT, "(구간 0.466–0.809)"),
    Check("p3val.conv.w_precision_hi", "conv ×1 층 가중 precision 구간 상한", 0.8092, 0.0006, "", _p3_val_ci("azure_conv_x1", "w_precision", 1), FIXEDSPLIT, "(구간 0.466–0.809)"),
    Check("p3val.code.w_meas_frac_pass2", "code ×16: 2회차 실측 ≥ 1.25 step 비율", 0.2353, 0.0006, "", _p3_val("azure_code_x16", "w_meas_frac", 2), FIXEDSPLIT, "| code ×16: regret ≥ 1.25 step 비율 | 14.1% | 23.5% (12.5–37.2%) | 23.5% | 19.6% |"),
    Check("p3val.code.w_meas_frac_keysteps", "code ×16: 키 step 수 가중 실측 ≥ 1.25 step 비율", 0.1959, 0.0006, "", _p3_val("azure_code_x16", "w_meas_frac", 1, "key_steps"), FIXEDSPLIT, "| code ×16: regret ≥ 1.25 step 비율 | 14.1% | 23.5% (12.5–37.2%) | 23.5% | 19.6% |"),
    Check("p3val.code.att_pred", "code ×16: 예측 attention 시간 비", 1.1345, 0.0005, "x", _p3_val("azure_code_x16", "w_pred_attention_ratio"), FIXEDSPLIT, "| code ×16: attention 시간 비 | 1.134 | 1.161 | 1.161 | 1.150 |"),
    Check("p3val.code.att_pass2", "code ×16: 2회차 실측 attention 시간 비", 1.1614, 0.0005, "x", _p3_val("azure_code_x16", "w_meas_attention_ratio", 2), FIXEDSPLIT, "| code ×16: attention 시간 비 | 1.134 | 1.161 | 1.161 | 1.150 |"),
    Check("p3val.code.att_keysteps", "code ×16: 키 step 수 가중 실측 attention 시간 비", 1.1498, 0.0005, "x", _p3_val("azure_code_x16", "w_meas_attention_ratio", 1, "key_steps"), FIXEDSPLIT, "| code ×16: attention 시간 비 | 1.134 | 1.161 | 1.161 | 1.150 |"),
    *[Check(f"p3val.conv.stratum.{s}.{c}", f"conv ×1 층 {s}: {t}", exp, 0.0006, "", _p3_val_stratum("azure_conv_x1", s, c), FIXEDSPLIT, txt)
      for s, c, t, exp, txt in [
          ("<1.1", "abs_err_median", "|예측−실측| 중앙값", 0.0355, "0.036 / 0.202"), ("<1.1", "abs_err_max", "|예측−실측| 최대", 0.2023, "0.036 / 0.202"),
          ("1.1-1.25", "abs_err_median", "|예측−실측| 중앙값", 0.0233, "0.023 / 0.099"), ("1.1-1.25", "abs_err_max", "|예측−실측| 최대", 0.0994, "0.023 / 0.099"),
          ("1.25-1.5", "abs_err_median", "|예측−실측| 중앙값", 0.0562, "0.056 / 0.112"), ("1.25-1.5", "abs_err_max", "|예측−실측| 최대", 0.1117, "0.056 / 0.112"),
          (">=1.5", "abs_err_median", "|예측−실측| 중앙값", 0.1401, "0.140 / 0.245"), (">=1.5", "abs_err_max", "|예측−실측| 최대", 0.2452, "0.140 / 0.245"),
          ("<1.1", "pred_median", "예측 regret 중앙값", 1.0, "| < 1.1 | 25 | 1.000 | 1.078 | 0 |"), ("1.1-1.25", "pred_median", "예측 regret 중앙값", 1.1660, "| 1.1–1.25 | 25 | 1.166 | 1.186 | 0 |"),
          ("1.1-1.25", "meas_median", "실측 regret 중앙값", 1.1862, "| 1.1–1.25 | 25 | 1.166 | 1.186 | 0 |")]],
    Check("p3trace.predicted_heuristic_us", "모델 예측: 휴리스틱 attention µs/층(시나리오 replay)", 180.87, 0.005, "us", _trace_predicted("predicted_attention_us_per_step", "heuristic"), FIXEDSPLIT, "예측 휴리스틱 180.9 µs/층"),
    Check("p3trace.model.r0_speedup", "model 반복 0의 TPOT 배율(휴리스틱 반복 0 / model 반복 0)", 0.8617, 0.0005, "x",
          lambda repo, data: float(_p3_repeats(data, TRACE_RUN, TRACE_NAME).loc[("heuristic", 0), "tpot_ms"] / _p3_repeats(data, TRACE_RUN, TRACE_NAME).loc[("model", 0), "tpot_ms"]), FIXEDSPLIT, "배율 0.86"),
    *[Check(f"p3trace.{pol}.{col}", f"트레이스 e2e {pol}: {col}", exp, tol, "", _p3_serve(TRACE_RUN, TRACE_NAME, pol, col), FIXEDSPLIT, txt)
      for pol, col, exp, tol, txt in [
          ("table", "tpot_ms_mean", 36.327, 0.0005, "| table | 5.893 | 36.327 |"), ("hybrid", "tpot_ms_mean", 38.779, 0.0005, "| hybrid(δ=0.2) | 5.877 | 38.779 |"),
          ("model", "tpot_ms_mean", 38.707, 0.0005, "| model | 5.800 | 38.707 |"), ("hybrid", "attn_ms_per_step", 5.877, 0.0005, "| hybrid(δ=0.2) | 5.877 | 38.779 |"),
          ("hybrid", "tpot_ms_p50", 38.856, 0.0005, "38.86 / 48.94"), ("hybrid", "tpot_ms_p90", 48.94, 0.0005, "38.86 / 48.94"),
          ("model", "tpot_ms_p50", 38.817, 0.0005, "38.82 / 49.01"), ("model", "tpot_ms_p90", 49.007, 0.0005, "38.82 / 49.01"),
          ("hybrid", "throughput_tokens_s", 756.7, 0.05, "| 756.7 |"), ("model", "throughput_tokens_s", 758.6, 0.05, "| 758.6 |"),
          ("table", "step_ms_per_step", 20.297, 0.0005, "step 시간 21.3 ms에서 20.3 ms"), ("heuristic", "step_ms_per_step", 21.302, 0.0005, "step 시간 21.3 ms에서 20.3 ms"),
          ("table", "speedup_ci95_low", 1.0341, 0.0005, "구간 1.034–1.035"), ("table", "speedup_ci95_high", 1.0354, 0.0005, "구간 1.034–1.035")]],
    *[Check(f"p3vllmx.{run}.{col}", f"vLLM 0.31 {run}: {col}", exp, 0.005, "ms", _p3_vllm_value(run, col), FIXEDSPLIT, txt)
      for run, col, exp, txt in [
          ("flash_attn_cudagraph_ragged", "gpu_busy_ms", 77.74, "| FA2 CUDA graph, 혼합 | 80.6 | 77.7 |"), ("flash_attn_cudagraph_ragged", "gemm_ms", 9.43, "| 77.7 | 67.7 | 9.4 |"),
          ("flash_attn_cudagraph_ragged", "not_gpu_busy_ms", 2.83, "| 9.4 | 0.6 | 2.8 |"), ("flash_attn_cudagraph_uniform", "attention_ms", 2.87, "| FA2 CUDA graph, 균일 | 13.7 | 13.0 | 2.9 |"),
          ("flash_attn_cudagraph_uniform", "gpu_busy_ms", 12.97, "| FA2 CUDA graph, 균일 | 13.7 | 13.0 | 2.9 |"), ("flash_attn_cudagraph_uniform", "gemm_ms", 9.5, "| 2.9 | 9.5 | 0.6 | 0.7 |"),
          ("flash_attn_cudagraph_uniform", "not_gpu_busy_ms", 0.72, "| 2.9 | 9.5 | 0.6 | 0.7 |"), ("flash_attn_eager_uniform", "gpu_busy_ms", 13.45, "| FA2 eager, 균일 | 14.6 | 13.4 |"),
          ("flash_attn_eager_uniform", "not_gpu_busy_ms", 1.18, "| 0.8 | 1.2 |"), ("flashinfer_eager_ragged", "gpu_busy_ms", 34.22, "| FlashInfer(tensor core) eager, 혼합 | 36.0 | 34.2 |"),
          ("flashinfer_eager_ragged", "gemm_ms", 9.56, "| 24.0 | 9.6 | 0.6 | 1.7 |"), ("flashinfer_eager_ragged", "not_gpu_busy_ms", 1.73, "| 24.0 | 9.6 | 0.6 | 1.7 |"),
          ("flashinfer_eager_uniform", "gpu_busy_ms", 13.27, "| FlashInfer eager, 균일 | 14.3 | 13.3 |"), ("flashinfer_eager_uniform", "gemm_ms", 9.66, "| 2.96 | 9.7 | 0.65 | 1.1 |"),
          ("flashinfer_eager_uniform", "not_gpu_busy_ms", 1.06, "| 2.96 | 9.7 | 0.65 | 1.1 |"), ("flash_attn_eager_ragged", "kv_write_ms", 0.07, "| 67.7 | 9.6 | 0.9 | 2.8 |")]],
    Check("p3vllm.other.fa2_eager_ragged", "vLLM FA2 eager 혼합: 기타(KV 쓰기 + 샘플링 등) ms", 0.89, 0.005, "ms",
          lambda repo, data: float(_p3_vllm(data)["table"]["flash_attn_eager_ragged"]["other_ms"] + _p3_vllm(data)["table"]["flash_attn_eager_ragged"]["kv_write_ms"]), FIXEDSPLIT, "| 67.7 | 9.6 | 0.9 | 2.8 |"),
    Check("p3vllm.steps_estimate", "FA2 eager 혼합 실행의 decode step 수 추정(커널 합계 / step당 커널 수; 정수 22이어야 함)", 22.0, 0.05, "",
          lambda repo, data: float(_p3_vllm(data)["table"]["flash_attn_eager_ragged"]["steps"]), FIXEDSPLIT, "step 수(약 22)"),
    Check("p3vllm.ratio.fa2_eager", "FA2 eager 혼합 / 균일 비프로파일 step 시간", 5.54, 0.005, "x",
          lambda repo, data: float(_p3_vllm(data)["table"]["flash_attn_eager_ragged"]["unprofiled_ms"] / _p3_vllm(data)["table"]["flash_attn_eager_uniform"]["unprofiled_ms"]), FIXEDSPLIT, "5.5배"),
    Check("p3vllm.ratio.fa2_graph", "FA2 CUDA graph 혼합 / 균일 비프로파일 step 시간", 5.89, 0.005, "x",
          lambda repo, data: float(_p3_vllm(data)["table"]["flash_attn_cudagraph_ragged"]["unprofiled_ms"] / _p3_vllm(data)["table"]["flash_attn_cudagraph_uniform"]["unprofiled_ms"]), FIXEDSPLIT, "5.9배"),
    Check("p3vllm.ratio.flashinfer", "FlashInfer 혼합 / 균일 비프로파일 step 시간", 2.51, 0.005, "x",
          lambda repo, data: float(_p3_vllm(data)["table"]["flashinfer_eager_ragged"]["unprofiled_ms"] / _p3_vllm(data)["table"]["flashinfer_eager_uniform"]["unprofiled_ms"]), FIXEDSPLIT, "2.5배"),
    Check("p3vllm.grid_heuristic_us", "격자의 같은 배치(B=32, 32768 + 512×31) FA2 휴리스틱 시간(µs/층)", 1071.8, 0.05, "us",
          lambda repo, data: float(pd.read_csv(data / "dispatch_paged_cold.csv").set_index("lens").loc["32768+512x31", "heuristic_us"]), FIXEDSPLIT, "층당 1,071.8 µs"),
    Check("p3text.ci.table.low", "64K 자연어 table 배율 구간 하한", 2.4474, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "table", "speedup_ci95_low"), FIXEDSPLIT, "2.449 (2.447–2.451)"),
    Check("p3text.ci.table.high", "64K 자연어 table 배율 구간 상한", 2.4506, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "table", "speedup_ci95_high"), FIXEDSPLIT, "2.449 (2.447–2.451)"),
    Check("p3text.ci.model.low", "64K 자연어 model 배율 구간 하한", 2.4359, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "model", "speedup_ci95_low"), FIXEDSPLIT, "2.443 (2.436–2.448)"),
    Check("p3text.ci.model.high", "64K 자연어 model 배율 구간 상한", 2.4476, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "model", "speedup_ci95_high"), FIXEDSPLIT, "2.443 (2.436–2.448)"),
    Check("p3text.ci.hybrid.low", "64K 자연어 hybrid 배율 구간 하한", 2.4318, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "hybrid", "speedup_ci95_low"), FIXEDSPLIT, "2.440 (2.432–2.449)"),
    Check("p3text.ci.hybrid.high", "64K 자연어 hybrid 배율 구간 상한", 2.4487, 0.0005, "x", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "hybrid", "speedup_ci95_high"), FIXEDSPLIT, "2.440 (2.432–2.449)"),
    Check("p3text.heuristic.tpot", "64K 자연어 휴리스틱 TPOT(ms)", 94.809, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "heuristic", "tpot_ms_mean"), FIXEDSPLIT, "| heuristic | 70.45 | 94.81 |"),
    Check("p3text.table.tpot", "64K 자연어 table TPOT(ms)", 38.714, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "table", "tpot_ms_mean"), FIXEDSPLIT, "| table | 14.33 | 38.71 |"),
    Check("p3text.model.tpot", "64K 자연어 model TPOT(ms)", 38.803, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "model", "tpot_ms_mean"), FIXEDSPLIT, "| model | 14.34 | 38.80 |"),
    Check("p3text.hybrid.tpot", "64K 자연어 hybrid TPOT(ms)", 38.853, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "hybrid", "tpot_ms_mean"), FIXEDSPLIT, "| hybrid | 14.34 | 38.85 |"),
    Check("p3text.model.attn", "64K 자연어 model attention ms/step", 14.336, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "model", "attn_ms_per_step"), FIXEDSPLIT, "| model | 14.34 | 38.80 |"),
    Check("p3text.hybrid.attn", "64K 자연어 hybrid attention ms/step", 14.336, 0.0005, "ms", _p3_serve(LONGTEXT_RUN, "demo_text_ragged_64k", "hybrid", "attn_ms_per_step"), FIXEDSPLIT, "| hybrid | 14.34 | 38.85 |"),
    Check("p3text.distinct_generations", "64K 자연어: 32개 요청 중 서로 다른 생성 수(휴리스틱 반복 0)", 30, 0, "",
          lambda repo, data: float(pd.read_parquet(data / "serve_4090" / LONGTEXT_RUN / "demo_text_ragged_64k" / "heuristic" / "repeat_000" / "tokens.parquet")
                                   .sort_values(["rid", "position"]).groupby("rid").token.apply(lambda g: tuple(g.astype(int))).nunique()), FIXEDSPLIT, "서로 다른 생성이 30개"),
    *[Check(f"p3gsm.attn.{pol}", f"GSM8K {pol}: attention ms/step", exp, 0.005, "ms", _p3_serve(GSM8K_RUN, "gsm8k_ragged32k_n200", pol, "attn_ms_per_step"), FIXEDSPLIT, txt)
      for pol, exp, txt in [("table", 7.679, "| table | 2.101 | 7.68 |"), ("fixed8", 7.403, "| 고정 8 | 2.154 | 7.40 |"), ("flashinfer_cudacore", 7.016, "| flashinfer_cudacore | 2.155 | 7.02 |")]],
    Check("p3gsm.marker_present", "GSM8K: `####`가 생성 안에 나온 문항 비율(휴리스틱)", 0.83, 0.0005, "", lambda repo, data: float(_p3_gsm8k(data).pred_marker_heuristic.notna().mean()), FIXEDSPLIT, "83%(200문항 중 166)"),
    Check("p3gsm.filler.fixed8_first_position_max", "GSM8K 고정 8: 필러 요청 분기 위치의 최댓값", 253, 0, "",
          lambda repo, data: float(__import__("kernelscope.serve.divergence", fromlist=["x"]).campaign_events(data / "serve_4090" / GSM8K_RUN).query("policy == 'fixed8' and repeat == 0").first_position.max()), FIXEDSPLIT, "고정 8 3~253"),
    Check("p3gsm.filler.first_position_min", "GSM8K table: 필러 요청 분기 위치의 최솟값", 3, 0, "",
          lambda repo, data: float(__import__("kernelscope.serve.divergence", fromlist=["x"]).campaign_events(data / "serve_4090" / GSM8K_RUN).query("policy == 'table' and repeat == 0").first_position.min()), FIXEDSPLIT, "table 3~145"),
    Check("p3gsm.filler.first_position_max", "GSM8K table: 필러 요청 분기 위치의 최댓값", 145, 0, "",
          lambda repo, data: float(__import__("kernelscope.serve.divergence", fromlist=["x"]).campaign_events(data / "serve_4090" / GSM8K_RUN).query("policy == 'table' and repeat == 0").first_position.max()), FIXEDSPLIT, "table 3~145"),

    Check("bg.len.conv.p50", "발표 배경: Azure conv 프롬프트 길이 중앙값(19,366건)", 1020, 0, "tokens", _bg_len("Azure conv", "p50"), BG, "conv 중앙값 1,020"),
    Check("bg.len.conv.p99", "발표 배경: Azure conv 프롬프트 길이 p99", 4142, 0.5, "tokens", _bg_len("Azure conv", "p99"), BG, "p99 4,142"),
    Check("bg.len.conv.max", "발표 배경: Azure conv 프롬프트 길이 최댓값", 14050, 0, "tokens", _bg_len("Azure conv", "max"), BG, "최대 14,050"),
    Check("bg.len.conv.n_ge_8192", "발표 배경: Azure conv에서 프롬프트 8K 이상인 요청 수", 1, 0, "requests", _bg_len("Azure conv", "n_ge_8192"), BG, "8K 이상은 conv 1건"),
    Check("bg.len.code.p50", "발표 배경: Azure code 프롬프트 길이 중앙값(8,819건)", 1469, 0, "tokens", _bg_len("Azure code", "p50"), BG, "code 중앙값 1,469"),
    Check("bg.len.burst.p50", "발표 배경: BurstGPT 7일 프롬프트 길이 중앙값(17,893건)", 502, 0, "tokens", _bg_len("BurstGPT", "p50"), BG, "BurstGPT 중앙값 502"),
    Check("bg.len.all.n_ge_32768", "발표 배경: 세 트레이스 합계에서 프롬프트 32K 이상인 요청 수", 0, 0, "requests", _bg_len_all("n_ge_32768"), BG, "32K 이상 0건"),
    Check("bg.kernel.fa2_heuristic_max", "발표 배경: 32K 격자 혼합 길이 FA2 휴리스틱 / 전체 최선 손실 최댓값(162셀)", 4.106, 0.0005, "x", _defaults_summary("32k", "ragged", "fa2_heuristic", "loss_max"), BG, "4.106배"),
    Check("bg.cta.heuristic_ctas", "발표 배경: 대표 셀(32768+512×31) 휴리스틱 launch의 CTA 수(분할 1)", 256, 0, "CTAs", _bg_cta("flashdecoding_paged", "ctas"), BG, "CTA 256개"),
    Check("bg.cta.heuristic_max_keys", "발표 배경: 같은 launch에서 가장 큰 CTA의 KV 토큰 수", 32768, 0, "keys", _bg_cta("flashdecoding_paged", "max"), BG, "한 CTA가 KV 32,768개"),
    Check("bg.cta.heuristic_max_over_sm_share", "발표 배경: 가장 큰 CTA / SM당 균등 몫(총 작업량 ÷ 128)", 10.78, 0.005, "x", _bg_cta("flashdecoding_paged", "max_over_sm_share"), BG, "10.8배"),
    Check("bg.cta.s16_ctas", "발표 배경: 분할 16 launch의 CTA 수", 4096, 0, "CTAs", _bg_cta("fd_s16_paged", "ctas"), BG, "CTA 4,096개"),
    Check("bg.cta.s16_max_keys", "발표 배경: 분할 16 launch에서 가장 큰 CTA의 KV 토큰 수", 2048, 0, "keys", _bg_cta("fd_s16_paged", "max"), BG, "최대 2,048"),
]


def run_checks(checks, repo: Path, data: Path) -> pd.DataFrame:
    repo, data = Path(repo), Path(data)
    rows = []
    for c in checks:
        measured, note = math.nan, ""
        try:
            measured = float(c.compute(repo, data))
            status = "PASS" if abs(measured - c.expected) <= c.tol else "FAIL"
        except (FileNotFoundError, KeyError, StopIteration, IndexError) as e:
            status, note = "MISSING", f"{type(e).__name__}: {e}"
        if status == "PASS" and c.doc:
            path = repo / c.doc
            if not path.exists() or c.doc_text not in path.read_text():
                status, note = "DOC_DRIFT", f"{c.doc} no longer states {c.doc_text!r}"
        rows.append({"id": c.id, "title": c.title, "expected": c.expected, "measured": measured, "tol": c.tol,
                     "unit": c.unit, "status": status, "doc": c.doc or "", "note": note})
    return pd.DataFrame(rows)
