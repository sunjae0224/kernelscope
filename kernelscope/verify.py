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
