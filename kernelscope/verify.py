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

CHECKS = [
    Check("kernel.uniform_median", "균일 배치(S1, dense, cold): 기본 휴리스틱의 최적 대비 손실 중앙값",
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
