#!/usr/bin/env bash
# KernelScope 시연 실행기. 프로젝트 루트에서 `./run.sh <명령>`. 인자 없이 실행하면 사용법을 보여 준다.
#
#   GPU 없이:   check · dashboard · verify [--tests]
#   GPU 필요:   kernel · serve · diagnose · divergence · generate "<프롬프트>" [정책] · gpu-all
#
# GPU 단계는 먼저 `serve doctor`로 모델·flash-attn·GPU 점유를 확인하고, 다른 GPU 프로세스가 있으면 기다리지 않고
# 그 이유를 출력한 뒤 멈춘다(기록을 오염시키지 않기 위해). 결과는 리포 밖 $KERNELSCOPE_RESULTS/demo_runs/<시각>/에 쌓인다.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${KERNELSCOPE_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
if [[ -z "${KERNELSCOPE_RESULTS+x}" ]]; then
  if [[ -d "$ROOT/../kernelscope/results" ]]; then KERNELSCOPE_RESULTS="$ROOT/../kernelscope/results"; else KERNELSCOPE_RESULTS="$ROOT/results"; fi
fi
mkdir -p "$KERNELSCOPE_RESULTS"; KERNELSCOPE_RESULTS="$(cd -- "$KERNELSCOPE_RESULTS" && pwd)"
export KERNELSCOPE_RESULTS HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
STAMP="${DEMO_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${DEMO_OUT:-$KERNELSCOPE_RESULTS/demo_runs/$STAMP}"
MODEL="${DEMO_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TABLE=demo_data/dispatch_paged_cold.csv
HYBRID="hybrid:$TABLE:0.2"
MODEL_INPUTS=(--machine machines/rtx4090.json --params models/rtx4090.json)
WORST=decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

usage() {
  cat <<'USAGE'
KernelScope 시연 실행기 — ./run.sh <명령>

  GPU 없이 (노트북에서도 됨)
    check        환경 점검: 파이썬·패키지·기록 번들·GPU 상태(있으면)
    dashboard    저장된 실측으로 대시보드 실행 → http://localhost:8501  (PORT=8502 ./run.sh dashboard)
    verify       문서의 수치 111개를 원본 기록에서 다시 계산해 대조 (--tests 를 붙이면 CPU 테스트도)

  GPU 필요 (연구실 RTX 4090, 다른 GPU 프로세스가 없을 때)
    kernel       최악 셀(32K×1 + 512×31, 페이지 KV) 하나에서 커널 변형 비교: 휴리스틱 / 분할 8·16 / FlashInfer   ~30초
    serve        Qwen3-4B 혼합 길이 실생성: heuristic · table · hybrid 정책의 TPOT와 토큰 일치 (REPEATS=1)       ~1.5분
    diagnose     decode step을 8개 연산 클래스로 분해해 attention 비중과 상한 판정                              ~1분
    divergence   교사 강제 진단으로 serve 결과의 토큰 분기 사건을 tie_1ulp/tie_2ulp/clear 로 분류                ~1분
    generate "<프롬프트>" [정책]   실제 텍스트 생성 1건 (정책: heuristic | table | hybrid | model)
    demo-record  대시보드 레이스용 자연어 혼합 길이 기록(demo_text_ragged: 32K×1 + 512×31, 64토큰): heuristic · table · hybrid, REPEATS=3  ~2분
    gpu-all      kernel → serve → diagnose → divergence 를 한 번에 (같은 결과 폴더)

  시연 순서 제안
    1) ./run.sh check  2) ./run.sh dashboard (첫 화면 Demo: 결론 → 경주 → 왜 → 어떻게 → 검증; Q&A는 Lab 페이지의 01~05 탭)  3) 시간이 되면 ./run.sh gpu-all 로 라이브 측정
    4) ./run.sh verify 로 "문서의 숫자는 기록에서 다시 계산된다"를 보여 주고 마무리

  환경 변수: KERNELSCOPE_RESULTS(결과 루트), DEMO_OUT(이번 시연 결과 폴더), REPEATS(serve 반복 기본 1 · demo-record 기본 3), PORT(대시보드)
USAGE
}

DOCTOR_DONE=0
doctor() {  # GPU 단계 공통 관문(한 번만). 실패 이유를 그대로 보여 주고 멈춘다.
  [[ "$DOCTOR_DONE" -eq 1 ]] && return 0
  DOCTOR_DONE=1
  say "serve doctor — 모델·flash-attn·GPU 점유 확인"
  if ! "$PY" -m kernelscope.cli serve doctor --model "$MODEL" > "$OUT/doctor.json" 2>&1; then
    "$PY" - "$OUT/doctor.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("   준비 안 됨:"); [print("    -", e) for e in d.get("errors", [])]
except Exception:
    print(open(sys.argv[1]).read()[-2000:])
PYEOF
    echo "   다른 GPU 프로세스가 끝난 뒤 다시 실행하세요 (nvidia-smi 로 확인)."; exit 1
  fi
  "$PY" - "$OUT/doctor.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"   ready: {d['ready']} · {d.get('gpu_name')} · free {d.get('free_memory_bytes', 0)/2**30:.1f} GiB · torch {d['packages']['torch']} · flash-attn {d['packages']['flash-attn']}")
PYEOF
}

cmd_check() {
  say "환경"
  note "python: $PY"
  "$PY" - <<'PYEOF'
import importlib, sys
print(f"   python {sys.version.split()[0]}")
for name in ("torch", "flash_attn", "flashinfer", "pandas", "streamlit"):
    try:
        m = importlib.import_module(name); print(f"   {name:10s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"   {name:10s} 없음 ({type(e).__name__})")
try:
    import torch; print(f"   cuda available: {torch.cuda.is_available()}" + (f" · {torch.cuda.get_device_name()}" if torch.cuda.is_available() else ""))
except Exception: pass
PYEOF
  say "기록 번들"
  for d in demo_data/hw_4090/ragged_s1_paged demo_data/serve_4090/graduation_20260922 demo_data/serve_4090/hybrid_20261002 demo_data/serve_4090/diagnose_20260927 demo_data/hw_4090/ragged_s1_flashinfer_cudacore; do
    if [[ -d "$d" ]]; then note "있음  $d"; else note "없음  $d"; fi
  done
  note "결과 루트: $KERNELSCOPE_RESULTS"
  if command -v nvidia-smi >/dev/null 2>&1; then
    say "GPU"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | sed 's/^/   /'
    busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -vi rerun || true)
    if [[ -n "$busy" ]]; then note "다른 GPU 프로세스(측정 전에 끝나야 함):"; echo "$busy" | sed 's/^/     /'; else note "다른 GPU 프로세스 없음"; fi
  else
    note "nvidia-smi 없음 — GPU 단계는 이 장비에서 돌릴 수 없고 dashboard·verify 만 가능"
  fi
  if [[ -x "$HOME/.venvs/kernelscope-vllm/bin/python" ]]; then note "vLLM 전용 환경 있음: ~/.venvs/kernelscope-vllm (scripts/vllm_reproduce.py)"; fi
}

cmd_dashboard() { say "대시보드 (Ctrl+C 로 종료) → http://localhost:${PORT:-8501}"; exec bash scripts/demo.sh; }

cmd_verify() {
  say "kernelscope verify — 문서 수치 재계산 (GPU 불필요)"
  "$PY" -m kernelscope.cli verify
  if [[ "${1:-}" == "--tests" ]]; then say "CPU 테스트"; CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider -m 'not gpu'; fi
}

cmd_kernel() {
  mkdir -p "$OUT"; doctor
  say "커널 변형 비교 — $WORST (cold, 페이지 KV)"
  "$PY" -m kernelscope.cli bench --workload "$WORST" --plugins flashdecoding_paged,fd_s8_paged,fd_s16_paged,flashinfer_paged_cudacore \
    --results "$OUT/kernel" --cache-state cold --warmup 5 --iters 20 > "$OUT/kernel.log" 2>&1 || { tail -n 20 "$OUT/kernel.log"; exit 1; }
  "$PY" - "$OUT/kernel/summaries.jsonl" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok = {r["plugin"]: r["kernel_time_us"] for r in rows if r.get("status") == "ok"}
base = ok.get("flashdecoding_paged")
labels = {"flashdecoding_paged": "라이브러리 휴리스틱 (기본)", "fd_s8_paged": "분할 8", "fd_s16_paged": "분할 16 (측정 테이블의 선택)", "flashinfer_paged_cudacore": "FlashInfer (CUDA-core)"}
print(f"   {'변형':34s} {'커널 시간(µs)':>14s} {'휴리스틱 대비':>12s}")
for name, us in ok.items():
    print(f"   {labels.get(name, name):34s} {us:14.1f} {base / us:11.2f}배")
bad = [r for r in rows if r.get("status") != "ok"]
for r in bad: print(f"   실패: {r['plugin']}: {str(r.get('error', ''))[:120]}")
PYEOF
  note "원본: $OUT/kernel/summaries.jsonl"
}

cmd_serve() {
  mkdir -p "$OUT"; doctor
  local repeats="${REPEATS:-1}"
  say "실생성 비교 — graduation_ragged (32K×1 + 512×31), heuristic · table · hybrid, ${repeats}회"
  note "warm-up 1회(2 step) 뒤 측정. 정책 캐시는 유지(keep): 첫 결정 비용은 warm-up이 흡수"
  if ! "$PY" -m kernelscope.cli serve run --model "$MODEL" --scenario scenarios/graduation_ragged.yaml \
      --policy heuristic --policy "table:$TABLE" --policy "$HYBRID" "${MODEL_INPUTS[@]}" \
      --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats "$repeats" --seed 0 --policy-cache keep \
      --out "$OUT/ragged" > "$OUT/serve.log" 2>&1; then
    if [[ -f "$OUT/ragged/summary.csv" ]]; then note "완료했지만 토큰 불일치가 기록됨 (아래 tokens_equivalent 열)"; else tail -n 20 "$OUT/serve.log"; exit 1; fi
  fi
  "$PY" - "$OUT/ragged/summary.csv" <<'PYEOF'
import sys, pandas as pd
s = pd.read_csv(sys.argv[1]).set_index("policy")
print(f"   {'정책':10s} {'TPOT(ms)':>9s} {'배율':>7s} {'attention/step(ms)':>19s} {'선택 비용(µs/step)':>18s}  토큰 일치")
for p in ("heuristic", "table", "hybrid"):
    if p not in s.index: continue
    r = s.loc[p]
    print(f"   {p:10s} {r.tpot_ms_mean:9.2f} {r.speedup_vs_heuristic:6.3f}배 {r.attn_ms_per_step:19.2f} {r.policy_us_per_step:18.1f}  {'예' if r.get('tokens_equivalent', True) else '아니오'}")
PYEOF
  note "원본: $OUT/ragged (manifest.json, 정책별 steps/tokens parquet)"
}

cmd_diagnose() {
  mkdir -p "$OUT"; doctor
  say "decode step 연산 분해 — graduation_ragged, heuristic vs table"
  "$PY" -m kernelscope.cli serve diagnose --model "$MODEL" --scenario scenarios/graduation_ragged.yaml \
    --policy heuristic --policy "table:$TABLE" --kv-gib 10 --warmup-runs 1 --warmup-steps 2 \
    --out "$OUT/diagnose/ragged" > "$OUT/diagnose.log" 2>&1 || { tail -n 20 "$OUT/diagnose.log"; exit 1; }
  "$PY" - "$OUT/diagnose/ragged/diagnosis.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for policy, p in d["policies"].items():
    att = next(r for r in p["ops"] if r["op_class"] == "attention")
    top = sorted(p["ops"], key=lambda r: -r["share"])[:4]
    print(f"   {policy:10s} attention 비중 {100*att['share']:.1f}% · 판정 {att['verdict']} · 상위 클래스: " + ", ".join(f"{r['op_class']} {100*r['share']:.1f}%" for r in top))
PYEOF
  note "그림: $OUT/diagnose/ragged/op_breakdown.png · 표: ops.csv"
}

cmd_divergence() {
  mkdir -p "$OUT"; doctor
  [[ -d "$OUT/ragged/heuristic" ]] || { note "먼저 ./run.sh serve (같은 DEMO_OUT) 를 실행해야 비교할 토큰 기록이 생깁니다."; exit 1; }
  say "교사 강제 진단 — 휴리스틱의 토큰 이력을 table·hybrid 에 넣고 로짓 비교"
  "$PY" -m scripts.check_policy_numerics --model "$MODEL" --scenario scenarios/graduation_ragged.yaml --reference heuristic \
    --policy "table:$TABLE" --policy "$HYBRID" "${MODEL_INPUTS[@]}" --steps 64 --logit-steps 64 --kv-gib 10 --seed 0 \
    --out "$OUT/numerics/ragged" > "$OUT/numerics.log" 2>&1 || { tail -n 20 "$OUT/numerics.log"; exit 1; }
  say "분기 사건 분류 (GPU 불필요)"
  "$PY" -m scripts.classify_divergence --campaign "$OUT" || note "조사 대상 사건이 있음 — 위 표와 $OUT/divergence.csv 참고"
}

cmd_generate() {
  local prompt="${1:-}"; local policy="${2:-heuristic}"
  [[ -n "$prompt" ]] || { echo "사용법: ./run.sh generate \"<프롬프트>\" [heuristic|table|hybrid|model]"; exit 1; }
  mkdir -p "$OUT"; doctor
  case "$policy" in table) policy="table:$TABLE";; hybrid) policy="$HYBRID";; esac
  say "텍스트 생성 — 정책 $policy"
  "$PY" -m kernelscope.cli serve generate --model "$MODEL" --prompt "$prompt" --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
    --policy "$policy" "${MODEL_INPUTS[@]}" --out "$OUT/generate_$(date +%H%M%S).json"
}

cmd_demo_record() {
  mkdir -p "$OUT"; doctor
  local repeats="${REPEATS:-3}"
  say "데모 레이스 기록 — demo_text_ragged (자연어, 32K×1 + 512×31, 64토큰), heuristic · table · hybrid, ${repeats}회, 정책 캐시 유지"
  "$PY" -m kernelscope.cli serve run --model "$MODEL" --scenario scenarios/demo_text_ragged.yaml \
      --policy heuristic --policy "table:$TABLE" --policy "$HYBRID" "${MODEL_INPUTS[@]}" \
      --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats "$repeats" --seed 0 --policy-cache keep \
      --out "$OUT/demo_text/ragged" > "$OUT/demo_record.log" 2>&1 || { tail -n 20 "$OUT/demo_record.log"; exit 1; }
  "$PY" scripts/export_race.py --run "$OUT/demo_text/ragged" --policies heuristic,table,hybrid
  note "대시보드 Demo의 '다른 기록 고르기'에서 $OUT/demo_text/ragged 를 고르면 글이 보이는 레이스가 재생됩니다."
  note "번들에 넣으려면: $PY scripts/package_demo.py --results $KERNELSCOPE_RESULTS"
}

case "${1:-help}" in
  check) cmd_check ;;
  dashboard) cmd_dashboard ;;
  verify) cmd_verify "${2:-}" ;;
  kernel) cmd_kernel ;;
  serve) cmd_serve ;;
  diagnose) cmd_diagnose ;;
  divergence) cmd_divergence ;;
  generate) cmd_generate "${2:-}" "${3:-heuristic}" ;;
  demo-record) cmd_demo_record ;;
  gpu-all) cmd_kernel; cmd_serve; cmd_diagnose; cmd_divergence; say "끝 — 결과 폴더: $OUT" ;;
  help|-h|--help) usage ;;
  *) echo "모르는 명령: $1"; usage; exit 1 ;;
esac
