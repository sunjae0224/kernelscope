#!/usr/bin/env bash
# KernelScope 터미널 시연 — 명령 하나. 프로젝트 루트에서 `./demo.sh` (옵션은 `./demo.sh --help`).
#
#   ./demo.sh                 64K 혼합 길이 기록 재생 (GPU 불필요, 15초 안팎)
#   ./demo.sh --scenario 32k  32K 혼합 길이
#   ./demo.sh --full          시뮬레이터의 시간별 활성 SM 수도 출력
#   ./demo.sh --step          장면마다 Enter 로 넘기기
#   ./demo.sh --live          연구실 RTX 4090에서 지금 측정한 뒤 재생 (1~3분)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${KERNELSCOPE_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8
exec "$PY" scripts/terminal_demo.py "$@"
