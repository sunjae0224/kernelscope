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
