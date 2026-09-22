"""Generate a reviewable report from completed, recorded serving experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernelscope.serve.report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = ["# Qwen3-4B 실제 생성 비교", "",
             "RTX 4090의 단일 프로세스 전체 모델 decode 실험. 합성 토큰 프롬프트로 길이를 통제했다. "
             "네트워크나 운영 서버 부하를 측정한 결과는 아니다.", "",
             "TPOT는 요청별 생성 토큰 간 간격의 평균이다. 첫 생성 토큰 뒤에 다른 요청의 prefill이 "
             "실행되면 그 대기도 포함한다. Decode wall은 정책 CPU 계산과 토큰 추출을 포함한 각 decode "
             "호출의 host 관측 시간이며 prompt admission은 제외한다. 이벤트 기반 step 시간과 구분한다.", ""]
    for scenario in sorted(p for p in args.campaign.iterdir() if p.is_dir()):
        path = scenario / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            lines += [f"## {scenario.name}", "", f"미완료 실험: {manifest.get('status')}", ""]
            continue
        table = summarize(scenario)
        if table.empty:
            continue
        lines += [f"## {scenario.name}", "",
                  f"모델: `{manifest['model']}` · 반복: {manifest['repeats']} · "
                  f"요청 수: {len(manifest['requests'])} · seed: {manifest['seed']}", "",
                  "| 정책 | Attention ms/step | Decode wall ms/step | 선택 µs/step | TPOT ms | TPOT 배속 | 95% 반복 bootstrap | 토큰 일치 |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
        for row in table.itertuples():
            ci = f"{row.speedup_ci95_low:.3f}–{row.speedup_ci95_high:.3f}"
            same = getattr(row, "tokens_equivalent", None)
            verdict = "일치" if same is True or str(same) == "True" else "불일치/미검증"
            lines.append(f"| {row.policy} | {row.attn_ms_per_step:.3f} | {row.decode_wall_ms_per_step:.3f} | "
                         f"{row.policy_us_per_step:.1f} | {row.tpot_ms_mean:.3f} | {row.speedup_vs_heuristic:.3f}× | "
                         f"{ci} | {verdict} |")
        lines += ["", "배속은 휴리스틱 TPOT / 후보 TPOT이다. 소수 반복의 bootstrap 구간은 이 실행 안의 "
                  "변동을 설명하며 다른 GPU·날짜·실제 요청 분포에 대한 일반화 보장은 아니다. "
                  "토큰 불일치가 있는 정책의 속도 차이는 출력 보존 개선으로 주장하지 않는다.", "",
                  f"원본: `{scenario.resolve()}`", ""]
    lines += ["## 재현", "", "```bash", "REPEATS=5 bash scripts/campaign.sh <새 결과 폴더>",
              f".venv/bin/python scripts/summarize_campaign.py <결과 폴더> --out {args.out}", "```", "",
              "각 manifest에는 모델 snapshot, 시나리오 해시, 패키지 버전, 정책 입력 해시, "
              "반복 순서와 GPU 사용 상태가 기록된다. 결과는 기존 디렉터리를 덮어쓰지 않는다.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(args.out)


if __name__ == "__main__":
    main()
