#!/usr/bin/env bash
# Start the read-only demo against measured artifacts. No GPU is needed.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_PYTHON="${KERNELSCOPE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$DEMO_PYTHON" ]]; then
  DEMO_PYTHON="$(command -v python3)"
fi
cd "$PROJECT_ROOT"
if [[ -z "${KERNELSCOPE_RESULTS+x}" ]]; then
  if [[ -d "$PROJECT_ROOT/../kernelscope/results" ]]; then
    export KERNELSCOPE_RESULTS="$PROJECT_ROOT/../kernelscope/results"
  else
    export KERNELSCOPE_RESULTS="$PROJECT_ROOT/demo_data"
  fi
fi
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$DEMO_PYTHON" -m streamlit run dashboard/app.py \
  --server.address 127.0.0.1 --server.port "${PORT:-8501}" \
  --server.headless true --browser.gatherUsageStats false "$@"
