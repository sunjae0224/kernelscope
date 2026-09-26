#!/usr/bin/env bash
# Build a .cu for GPGPU-Sim (PTX only, sm_70) and run it under the SM7_QV100 config in a scratch dir.
#   ./run_sim.sh <name.cu> <run-dir> [program args...]      env: FUNC=1 for functional-only mode
# Notes (CUDA 11.8 + GPGPU-Sim 4.2 dev a4ce3fe): the fatbin must be uncompressed and carry an identifier
# (-lineinfo) for cuobjdump extraction, and CUOBJDUMP_SIM_FILE=1 skips the pre-CUDA-6 section pruning
# that otherwise aborts with "No PTX sections found". Nothing in the simulator tree is modified.
set -eo pipefail
ROOT=/root/sunjae/gpu_graduation
SRC=$(readlink -f "$1"); RUN=$2; shift 2
export PATH=$ROOT/toolchain-gcc11:/usr/local/cuda-11.8/bin:$PATH
export CUDA_INSTALL_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$ROOT/gpgpu-sim_distribution/lib/gcc-11.5.0/cuda-11080/release:${LD_LIBRARY_PATH:-}
mkdir -p "$RUN" && cd "$RUN"
BIN=$(basename "${SRC%.cu}")
nvcc -ccbin g++-11 -O3 -gencode arch=compute_70,code=compute_70 -Xfatbin=-compress=false --cudart shared -o "$BIN" "$SRC"
cp -n $ROOT/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/gpgpusim.config $ROOT/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/config_volta_islip.icnt . 2>/dev/null || true
export CUOBJDUMP_SIM_FILE=1
[ "${FUNC:-0}" = 1 ] && export PTX_SIM_MODE_FUNC=1
stdbuf -oL ./"$BIN" "$@" > sim.log 2>&1 || { echo "run failed (exit $?)"; tail -5 sim.log; exit 1; }
grep -E '^gpu_sim_cycle =|^gpu_sim_insn =|^gpu_ipc =|^gpu_tot_sim_cycle|check' sim.log | tail -6
