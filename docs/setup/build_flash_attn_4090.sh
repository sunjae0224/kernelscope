#!/usr/bin/env bash
# Same upstream version as the requested wheel, rebuilt for Ubuntu 20.04 glibc.
# Build only; installation is a separate explicit command after the wheel exists.
set -euo pipefail
WORK=${ACCELSIM_WORK_ROOT:-/home/skkai/accelsim}
PREFIX=/home/skkai/miniforge3/envs/accelsim-build
PY=/home/skkai/miniforge3/envs/gradkernel/bin/python
SRC="$WORK/flash-attention"
mkdir -p "$WORK/logs" "$WORK/wheels"
if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 --branch v2.8.3.post1 --recursive --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git "$SRC"
fi
"$PY" -m pip install ninja packaging psutil
export PATH="$(dirname "$PY"):$PREFIX/bin:/usr/bin:/bin"
export CUDA_HOME="$PREFIX"
export CC="$PREFIX/bin/x86_64-conda-linux-gnu-cc" CXX="$PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CPATH="$PREFIX/targets/x86_64-linux/include"
export LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib"
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=${MAX_JOBS:-4} NVCC_THREADS=1
cd "$SRC"
/usr/bin/time -f 'flash-local-wheel wall_s=%e exit=%x' -a -o "$WORK/logs/timings.txt" \
    "$PY" -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$WORK/wheels" \
    > "$WORK/logs/flash-local-wheel.log" 2>&1
echo "Built wheels in $WORK/wheels; replace only flash-attn with the same-version local wheel."
