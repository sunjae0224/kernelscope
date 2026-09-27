"""Export provenance-separated follow-up results and publication-ready plots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernelscope.dashboard.research import campaign_overview, experiment_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads((args.results / "campaign.json").read_text())
    directories = [args.results / job["id"] for job in campaign["jobs"]]
    if any(job["status"] != "complete" for job in campaign["jobs"]):
        raise SystemExit("All prespecified experiments must complete before the final report")
    table = campaign_overview(directories)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "summary.csv", index=False)
    import pandas as pd
    from kernelscope.serve.report import iter_runs, tpot_us
    repeats = []
    for directory in directories:
        for policy, repeat, run, meta in iter_runs(directory):
            steps = pd.read_parquet(run / "steps.parquet")
            tokens = pd.read_parquet(run / "tokens.parquet")
            repeats.append({**experiment_identity(directory, meta), "policy": policy, "repeat": repeat,
                            "tpot_ms": float(tpot_us(tokens).dropna().mean() / 1000),
                            "decode_wall_ms": float(steps.decode_wall_us.mean() / 1000),
                            "attention_ms": float(steps.attn_us.mean() / 1000),
                            "policy_us": float(steps.policy_us.mean())})
    pd.DataFrame(repeats).to_csv(args.out / "repetitions.csv", index=False)
    comparisons = table[table.policy != "heuristic"]
    checked = int(comparisons.output_validation_passed.sum())
    protocol = campaign.get("warmup_protocol", "partial_scenario")
    warmup_note = ("각 정책으로 전체 시나리오를 한 번 워밍업했다. 측정마다 새 정책 객체를 만들어 첫 선택 비용을 보존했다."
                   if protocol == "full_scenario" else
                   "예비 측정은 요청당 decode 2스텝으로 워밍업을 제한했다. 요청 합류 시 실제 최대 배치를 워밍업하지 못하므로 최종 성능 근거에는 전체 워밍업 캠페인을 사용한다.")
    lines = ["# 자연어·두 모델 후속 검증", "",
             "RTX 4090 한 대에서 동일한 실측표·피팅 파라미터를 고정하고 두 모델, 세 시나리오, 두 시드를 측정했다. "
             "각 조합은 세 정책을 세 번 반복하며 실행 순서를 회전한다. 입력은 직접 작성한 문단으로 구성했으며 운영 트래픽이나 표준 언어 품질 벤치마크가 아니다.", "",
             warmup_note, "",
             "기존 측정 격자와 다른 배치 28, 짧은 문맥 384, 긴 문맥 12,288토큰을 사용했다. "
             "두 모델의 attention head 구성은 같으므로 다른 head 구성이나 다른 GPU로의 일반화는 검증하지 않는다.", "",
             "TPOT는 요청별 토큰 간 시간의 평균이며 도착 요청의 직렬 prefill과 정책 선택 비용을 포함한다. "
             "배율은 같은 모델·시나리오·시드·반복의 휴리스틱 TPOT ÷ 해당 정책 TPOT이다. "
             "생성 토큰 또는 스케줄이 다르면 배율을 성능 개선의 근거로 표시하지 않는다.", "",
             f"출력 검증을 통과한 정책·입력 조합은 **{checked}/{len(comparisons)}개**다. "
             "아래 표에는 불일치 사례도 모두 포함한다. 전체 캠페인의 종료 코드 1은 검증 불일치가 보존됐다는 뜻이며, "
             "완료 여부는 각 manifest의 `status`로 확인한다.", "",
             "| 모델 | 조건 | 시드 | 정책 | TPOT ms | 배율 | 출력 검증 |", "|---|---|---:|---|---:|---:|---|"]
    for row in table.itertuples():
        if row.policy == "heuristic":
            continue
        model = "Qwen 4B" if "Qwen" in row.model else "Llama 8B"
        speed = f"{row.speedup:.3f}×" if row.claim_eligible else "—"
        lines.append(f"| {model} | {row.scenario_family} | {row.seed} | {row.policy} | {row.tpot_ms:.3f} | {speed} | "
                     + ("일치" if row.output_validation_passed else "불일치/미검증") + " |")
    lines += ["", "## 정책 선택 비용", "",
              "아래 첫 호출·캐시 miss·hit는 서로 겹치지 않는 분류다. C 실행 경로의 컴파일/로드는 시작 단계에 수행하고 "
              "manifest의 `policy_setup_runs`에 따로 기록했다. 첫 호출에는 새 길이 구성의 전체 후보 예측이 포함된다.", "",
              "| 모델 | 조건 | 시드 | 첫 호출 µs | 후속 miss µs | hit µs |", "|---|---|---:|---:|---:|---:|"]
    for row in table[table.policy == "model"].itertuples():
        model = "Qwen 4B" if "Qwen" in row.model else "Llama 8B"
        def number(value):
            import math
            return f"{value:.1f}" if math.isfinite(value) else "—"
        lines.append(f"| {model} | {row.scenario_family} | {row.seed} | {number(row.first_decision_us)} | "
                     f"{number(row.cache_miss_us)} | {number(row.cache_hit_us)} |")
    lines += ["", "개별 반복의 시간은 [repetitions.csv](repetitions.csv), 반복 평균·표준편차, 95% paired bootstrap 구간, "
              "비교 불가 사유와 입력·소스 식별자는 [summary.csv](summary.csv)에 보존한다. "
              "이는 세 번 반복한 해당 실험의 변동이며 서비스 전체에 대한 보장 구간이 아니다.", "",
              "![모델과 시나리오별 TPOT](tpot.png)", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row", layout="constrained")
    models = list(table.model.unique())
    colors = {"heuristic": "#89949d", "table": "#087f8c", "model": "#7359b6"}
    for r, model in enumerate(models):
        for c, scenario in enumerate(("uniform", "ragged", "arrivals")):
            ax = axes[r, c]
            rows = table[(table.model == model) & (table.scenario_family == scenario)]
            for i, policy in enumerate(("heuristic", "table", "model")):
                selected = rows[rows.policy == policy].sort_values("seed")
                x = np.arange(len(selected)) + (i - 1) * .24
                bars = ax.bar(x, selected.tpot_ms, width=.23, label=policy, color=colors[policy],
                              yerr=selected.tpot_run_std_ms.fillna(0), capsize=2)
                for bar, valid in zip(bars, selected.output_validation_passed):
                    if not valid and policy != "heuristic":
                        bar.set_hatch("///")
                        bar.set_edgecolor("#7d1111")
            ax.set_title(("Qwen 4B" if "Qwen" in model else "Llama 8B") + " · " + scenario)
            ax.set_xticks([0, 1], ["seed 0", "seed 1"])
            ax.set_ylabel("Request-mean TPOT (ms)")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Controlled natural-text serving · RTX 4090 · 3 repeats (mean ± run SD)\nHatched bars: output equivalence failed; timing only")
    figure.savefig(args.out / "tpot.png", dpi=180)
    figure.savefig(args.out / "tpot.svg")
    plt.close(figure)
    print(f"Exported {len(table)} experiment/policy rows to {args.out}")


if __name__ == "__main__":
    main()
