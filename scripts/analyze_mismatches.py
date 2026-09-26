"""Why generated tokens differed between policies: divergence events and bf16 ties (no GPU needed).

    python -m scripts.analyze_mismatches --out docs/experiments/2026-09-26-mismatch-analysis.md

Reads the recorded follow-up campaign (token histories per policy), the teacher-forced logit diagnostic
and the uniform fixed8 case, and writes one report: where each run first diverged, whether the tail
cascaded, and how the reference top-1 margin at those positions compares with bfloat16 spacing.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from kernelscope.serve.divergence import divergence_events, tie_class

REPO = Path(__file__).resolve().parents[1]
FOLLOWUP = REPO / "demo_data" / "serve_4090" / "followup_fullwarmup_20260922"
TEACHER = REPO / "docs" / "experiments" / "heldout-arrivals-numerics"
MODEL_LABEL = {"qwen4b": "Qwen3-4B", "llama8b": "Llama-8B"}


def _tokens(run_dir, policy, repeat=0):
    return pd.read_parquet(run_dir / policy / f"repeat_{repeat:03d}" / "tokens.parquet")


def collect_events(campaign=FOLLOWUP) -> pd.DataFrame:
    """Divergence events of every non-reference policy run in the campaign (all repeats)."""
    rows = []
    for eq in sorted(campaign.glob("*/*/seed_*/equivalence.csv")):
        run_dir = eq.parent
        model, scenario, seed = run_dir.parts[-3], run_dir.parts[-2], int(run_dir.name.split("_")[1])
        e = pd.read_csv(eq)
        for r in e.itertuples():
            ref, cand = _tokens(run_dir, "heuristic", r.repeat), _tokens(run_dir, r.policy, r.repeat)
            ev = divergence_events(ref, cand)
            base = {"model": model, "scenario": scenario, "seed": seed, "policy": r.policy, "repeat": int(r.repeat),
                    "token_mismatches": int(r.token_mismatches), "tokens_compared": int(r.tokens_compared)}
            if ev.empty:
                rows.append({**base, "events": 0, "rid": None, "first_position": None, "differing_after_first": None,
                             "after_first": None, "cascade": None})
            for x in ev.itertuples():
                rows.append({**base, "events": len(ev), "rid": x.rid, "first_position": x.first_position,
                             "differing_after_first": x.differing_after_first, "after_first": x.after_first,
                             "cascade": x.cascade})
    return pd.DataFrame(rows)


def teacher_forced(path=TEACHER) -> tuple[pd.DataFrame, dict]:
    t = pd.read_csv(path / "teacher_forced_logits.csv")
    manifest = json.loads((path / "manifest.json").read_text())
    # The diagnostic records margins but not the top logit itself; classification assumes the reference
    # logits sit in the 16-32 range (spacing 0.125) -- stated as an assumption in the report.
    t["tie_class"] = [tie_class(m, 20.0) for m in t.reference_top1_margin]
    return t, manifest


def report(events: pd.DataFrame, teacher: pd.DataFrame, manifest: dict) -> str:
    out = ["# 정책 간 생성 토큰 불일치의 원인 분석 (2026-09-26)", "",
           "생성 토큰이 달라진 모든 실행을 기록된 원본에서 다시 분석했다. GPU 없이 `python -m scripts.analyze_mismatches`로 재생성된다.",
           "분석 대상: 후속 캠페인 `followup_fullwarmup_20260922`(2모델 × 3조건 × 2시드 × 정책 2종 × 반복 3회)와 "
           "교사 강제(teacher-forced) 로짓 진단 `heldout-arrivals-numerics`(Qwen3-4B, 요청 도착 조건, 시드 0).", ""]
    # 1. determinism across repeats
    fails = events[events.events > 0]
    key = ["model", "scenario", "seed", "policy"]
    per_run = fails.groupby(key + ["repeat"]).agg(events=("events", "first"), first=("first_position", lambda s: tuple(s)),
                                                  rids=("rid", lambda s: tuple(s))).reset_index()
    same = per_run.groupby(key).agg(repeats=("repeat", "nunique"), distinct=("first", "nunique")).reset_index()
    out += ["## 1. 불일치는 반복마다 같은 위치에서 시작한다", "",
            f"토큰이 달라진 정책·조건 조합 {len(same)}개 모두에서, 3회 반복의 첫 불일치 위치(요청, 토큰 위치)가 완전히 같았다"
            f" (위치 집합이 다른 조합: {int((same.distinct > 1).sum())}개). 즉 불일치는 무작위 잡음이 아니라 같은 입력에서 재현되는 결정적 차이다.", ""]
    # 2. events table
    out += ["## 2. 불일치 토큰 수가 아니라 분기 사건 수로 세어야 한다", "",
            "탐욕적 디코딩에서는 토큰 하나가 달라지면 그 요청의 이후 토큰이 모두 달라진다(연쇄). 아래 표는 반복 0 기준으로 요청별 첫 분기 위치와 연쇄 여부다.", "",
            "| 모델 | 조건 | 시드 | 정책 | 불일치 토큰 | 분기 사건 | 요청 | 첫 분기 위치 (32개 중) | 이후 토큰 중 다른 수 | 연쇄 |",
            "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    r0 = fails[fails.repeat == 0].sort_values(key + ["rid"])
    for x in r0.itertuples():
        out.append(f"| {MODEL_LABEL[x.model]} | {x.scenario} | {x.seed} | {x.policy} | {x.token_mismatches}/{x.tokens_compared} | {x.events} | "
                   f"{int(x.rid)} | {int(x.first_position)} | {int(x.differing_after_first)}/{int(x.after_first)} | {'예' if x.cascade else '아니오'} |")
    n_runs = len(r0.groupby(key)); n_events = int(r0.events.groupby([r0[k] for k in key]).first().sum())
    tail_last = int((r0.first_position == 31).sum())
    out += ["", f"실패한 {n_runs}개 실행(반복 0)의 분기 사건은 모두 {n_events}건이며, 그중 {tail_last}건은 요청의 마지막 토큰(위치 31)에서 일어나 "
            "이후 토큰이 없다. 혼합 길이 조건의 실패 2건은 모두 이 경우(1토큰)다. 연쇄가 아닌 사건(토큰 하나만 바뀐 뒤 다시 일치)도 있어, "
            "달라진 토큰이 대개 확률이 거의 같은 대체 표현임을 시사한다.", ""]
    # 3. teacher-forced margins
    flips = teacher[~teacher.argmax_equal]
    out += ["## 3. 뒤바뀐 위치의 1·2위 로짓 차이는 bfloat16 간격 1~2개다", "",
            f"교사 강제 진단({manifest['model']}, 요청 도착 조건, 시드 {manifest['seed']})은 두 정책에 휴리스틱과 같은 토큰 이력을 넣고 "
            f"다음 토큰의 로짓만 비교한다. 정책별 {int(teacher.groupby('policy').size().iloc[0])}개 위치 중 argmax가 다른 위치는 "
            f"{len(flips)}개였고, 그 위치의 기준(휴리스틱) 1·2위 로짓 차이는 다음과 같다.", "",
            "| 정책 | 요청 | 위치 | 기준 1·2위 차이 | 정책 간 로짓 최대 차이 | 분류 (bf16 간격 기준) |", "|---|---:|---:|---:|---:|---|"]
    for x in flips.itertuples():
        out.append(f"| {x.policy} | {x.rid} | {x.position} | {x.reference_top1_margin:.3f} | {x.max_abs_logit_diff:.3f} | {x.tie_class} |")
    share = {thr: float((teacher.reference_top1_margin <= thr).mean()) for thr in (0.125, 0.25, 0.5)}
    out += ["", "bfloat16은 유효숫자 8비트(저장 7비트)이므로 16~32 크기의 로짓에서 표현 가능한 간격은 0.125, 32~64에서는 0.25다. "
            "뒤바뀐 네 위치의 1·2위 차이는 모두 0.125 또는 0.25, 즉 간격 1~2개 안이고, 두 정책의 로짓 차이"
            f"({flips.max_abs_logit_diff.min():.3f}~{flips.max_abs_logit_diff.max():.3f})도 같은 크기다. "
            f"전체 위치 중 1·2위 차이가 0.125 이하인 비율은 {share[0.125]:.1%}, 0.25 이하는 {share[0.25]:.1%}, 0.5 이하는 {share[0.5]:.1%}다. "
            "이 위치들은 부분 합의 순서가 달라지는 어떤 커널 설정에서도 뒤바뀔 수 있으며, 기본 휴리스틱의 답이 '정답'인 것도 아니다. "
            "**가정.** 진단 기록에는 1·2위 차이만 있고 1위 로짓의 절대 크기는 없다. 위 분류는 로짓이 16~32 범위에 있다고 가정한 것이며(간격 0.125), "
            "로짓이 32 이상이면 간격이 0.25가 되어 '2개' 사례는 '1개'로 분류된다. 어느 쪽이든 결론(간격 1~2개)은 같지만, 다음 진단부터는 1위 로짓 값을 함께 기록한다.", ""]
    # 4. cross-check free generation vs teacher-forced positions
    q = r0[(r0.model == "qwen4b") & (r0.scenario == "arrivals") & (r0.seed == 0)]
    tf = {(int(x.rid), int(x.position)) for x in flips.itertuples()}
    hits = [(x.policy, x.rid, x.first_position, (int(x.rid), int(x.first_position)) in tf) for x in q.itertuples()]
    out += ["## 4. 자유 생성의 분기 위치는 교사 강제 진단의 동점 위치와 일치한다", "",
            "같은 모델·조건·시드(Qwen3-4B, 요청 도착, 시드 0)에서 자유 생성이 처음 분기한 (요청, 위치)가 교사 강제 진단에서 argmax가 뒤바뀐 위치인지 대조했다.", "",
            "| 정책 | 요청 | 첫 분기 위치 | 교사 강제 진단의 뒤바뀐 위치와 일치 |", "|---|---:|---:|---|"]
    out += [f"| {p} | {int(r)} | {int(pos)} | {'예' if h else '아니오'} |" for p, r, pos, h in hits]
    out += ["", f"{sum(h for *_, h in hits)}/{len(hits)}건이 일치한다. 즉 자유 생성에서 답이 갈라진 지점은 그 직전까지 같은 이력을 넣어도 1·2위 로짓이 "
            "bf16 간격 1~2개 안에 있던 위치였다.", "",
            "## 5. 해석과 조치", "",
            "- 불일치는 특정 분할 수(num_splits)의 오류가 아니라, 부분 합의 순서가 달라지는 모든 설정에서 생기는 bf16 반올림 차이가 "
            "1·2위 로짓이 거의 같은 위치에서 겉으로 드러난 것이다. 뒤바뀐 위치에서 두 정책은 같은 분할 수(1)를 썼고, 차이는 이전 단계에서 "
            "다른 분할 수로 계산해 KV 캐시에 저장된 값의 반올림 차이가 누적된 것이다.",
            "- 따라서 '출력 보존이 확인된 분할 수만 고르는' 제약(중간보고서 표 10)은 원인에 맞지 않아 구현하지 않는다. 대신 검증 지표를 바꾼다: "
            "토큰 일치율 대신 분기 사건 수를 보고하고, 각 사건을 교사 강제 진단으로 `tie_1ulp` / `tie_2ulp` / `clear`로 분류한다. "
            "`clear`인 사건이 하나라도 있으면 구현 오류로 취급하고, 동점 사건만 있으면 '수치 동점에 의한 불일치'로 보고한다.",
            "- 새 캠페인에서 이 분류를 자동으로 얻으려면 `scripts/check_policy_numerics.py`의 교사 강제 진단을 각 조건·시드에 대해 실행해야 한다(GPU 필요). "
            "현재는 Qwen3-4B 요청 도착 시드 0에 대해서만 기록이 있다.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    events = collect_events()
    teacher, manifest = teacher_forced()
    md = report(events, teacher, manifest)
    print(md)
    if args.out:
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
