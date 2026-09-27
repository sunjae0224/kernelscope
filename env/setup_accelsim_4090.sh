#!/usr/bin/env bash
# Run from any directory; no system installs or conda activation.
set -eo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA=/home/skkai/miniforge3/bin/conda
BUILD_PREFIX=${ACCELSIM_BUILD_PREFIX:-/home/skkai/miniforge3/envs/accelsim-build}
WORK=${ACCELSIM_WORK_ROOT:-/home/skkai/accelsim}
FRAMEWORK=${ACCELSIM_ROOT:-$WORK/accel-sim-framework}
mkdir -p "$WORK/logs"
step() {
    local name=$1; shift
    local rc=0
    /usr/bin/time -f "$name wall_s=%e exit=%x" -a -o "$WORK/logs/timings.txt" \
        "$@" > "$WORK/logs/$name.log" 2>&1 || rc=$?
    if [[ -n "${STEP_ARCHIVE:-}" ]]; then cp "$WORK/logs/$name.log" "$STEP_ARCHIVE/$name.log"; fi
    return "$rc"
}
build_env() {
    export PATH="$BUILD_PREFIX/bin:$PATH"
    export CC="$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    export CUDA_INSTALL_PATH="$BUILD_PREFIX" CUDA_HOME="$BUILD_PREFIX"
    export CPATH="$BUILD_PREFIX/targets/x86_64-linux/include:$BUILD_PREFIX/include:${CPATH:-}"
    export LIBRARY_PATH="$BUILD_PREFIX/targets/x86_64-linux/lib:$BUILD_PREFIX/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$BUILD_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    export ACCELSIM_CONFIG=release
}
case ${1:-help} in
toolchain)
    if [[ ! -x "$BUILD_PREFIX/bin/nvcc" ]]; then
        step conda-build "$CONDA" create -y -p "$BUILD_PREFIX" -c conda-forge -c nvidia \
            python=3.11 gcc_linux-64=11 gxx_linux-64=11 'cmake>=3.22,<4' \
            boost zstd zlib libxml2 openssl flex bison libgl-devel cuda-nvcc=12.9 cuda-cudart-dev=12.9 \
            cuda-cupti-dev=12.9 cuda-nvdisasm=12.9 cuda-cuobjdump=12.9 cuda-profiler-api=12.9
    fi
    "$CONDA" list -p "$BUILD_PREFIX" --explicit > "$WORK/logs/conda-build-explicit.txt"
    ;;
build)
    build_env
    if [[ ! -d "$FRAMEWORK/.git" ]]; then
        step clone git clone --branch v2.0.0 --depth 1 \
            https://github.com/accel-sim/accel-sim-framework.git "$FRAMEWORK"
    fi
    test "$(git -C "$FRAMEWORK" rev-parse HEAD)" = 64653015f85fb5664c84a10f48527e8897d289d0
    if git -C "$FRAMEWORK" apply --check "$REPO/docs/setup/accelsim-sm89.patch"; then
        git -C "$FRAMEWORK" apply "$REPO/docs/setup/accelsim-sm89.patch"
    else
        git -C "$FRAMEWORK" apply --reverse --check "$REPO/docs/setup/accelsim-sm89.patch"
    fi
    if [[ ! -d "$FRAMEWORK/gpu-simulator/gpgpu-sim/.git" ]]; then
        step clone-gpgpu git clone https://github.com/accel-sim/gpgpu-sim_distribution.git \
            "$FRAMEWORK/gpu-simulator/gpgpu-sim"
        git -C "$FRAMEWORK/gpu-simulator/gpgpu-sim" checkout 91880c53383d5a6a6742bfb1be2c5f34e39c7871
    fi
    test "$(git -C "$FRAMEWORK/gpu-simulator/gpgpu-sim" rev-parse HEAD)" = 91880c53383d5a6a6742bfb1be2c5f34e39c7871
    if [[ ! -e "$FRAMEWORK/gpu-simulator/extern/pybind11/.git" ]]; then
        step clone-pybind git clone https://github.com/pybind/pybind11.git \
            "$FRAMEWORK/gpu-simulator/extern/pybind11"
        git -C "$FRAMEWORK/gpu-simulator/extern/pybind11" checkout 296d5d1d3664dd0e0616e81920342e56b7249171
    fi
    test "$(git -C "$FRAMEWORK/gpu-simulator/extern/pybind11" rev-parse HEAD)" = 296d5d1d3664dd0e0616e81920342e56b7249171
    source "$FRAMEWORK/gpu-simulator/setup_environment.sh" release
    step configure "$BUILD_PREFIX/bin/cmake" -S "$FRAMEWORK/gpu-simulator" \
        -B "$FRAMEWORK/gpu-simulator/build" -DCMAKE_PREFIX_PATH="$BUILD_PREFIX" \
        -DPython_EXECUTABLE="$BUILD_PREFIX/bin/python" -DPython3_EXECUTABLE="$BUILD_PREFIX/bin/python" \
        -DPython_INCLUDE_DIR="$BUILD_PREFIX/include/python3.11" \
        -DPython_LIBRARY="$BUILD_PREFIX/lib/libpython3.11.so" \
        -DPython3_INCLUDE_DIR="$BUILD_PREFIX/include/python3.11" \
        -DPython3_LIBRARY="$BUILD_PREFIX/lib/libpython3.11.so"
    step simulator-build "$BUILD_PREFIX/bin/cmake" --build "$FRAMEWORK/gpu-simulator/build" -j16
    step simulator-install "$BUILD_PREFIX/bin/cmake" --install "$FRAMEWORK/gpu-simulator/build"
    cd "$FRAMEWORK/util/tracer_nvbit"
    if [[ ! -f nvbit_release/core/libnvbit.a ]]; then step nvbit-install bash install_nvbit.sh; fi
    step tracer-build make -C tracer_tool -j8 ARCH=sm_89 "CXX=$CXX"
    # The upstream postprocessor Makefile hardcodes system g++; compile explicitly.
    step postprocessor-build "$CXX" -std=c++17 -O3 -pthread \
        tracer_tool/traces-processing/post-traces-processing.cpp -lzstd \
        -o tracer_tool/traces-processing/post-traces-processing
    ;;
config)
    mkdir -p "$FRAMEWORK/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090" \
        "$FRAMEWORK/gpu-simulator/configs/tested-cfgs/SM89_RTX4090"
    cp "$REPO/docs/setup/SM89_RTX4090/gpgpusim.config" \
        "$FRAMEWORK/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config"
    cp "$REPO/docs/setup/SM89_RTX4090/trace.config" \
        "$FRAMEWORK/gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config"
    ;;
gate)
    build_env
    GATE="$WORK/gate-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$GATE"
    STEP_ARCHIVE="$GATE"
    # v2.0.0 accepts binary versions 80/86 but not 89. Ampere SASS runs on Ada.
    step vecadd-build "$BUILD_PREFIX/bin/nvcc" -ccbin "$CXX" -arch="${VECADD_ARCH:-sm_86}" \
        -O3 "$REPO/docs/setup/vecadd_4090.cu" -o "$GATE/vecadd"
    nvidia-smi > "$GATE/nvidia-smi.txt"
    cd "$GATE"
    step vecadd-native ./vecadd
    step vecadd-trace env CUDA_VISIBLE_DEVICES=0 DYNAMIC_KERNEL_RANGE='1@.*vecAdd.*' \
        ACTIVE_FROM_START=1 NVBIT_INSTRUMENTATION_ENABLED=1 TERMINATE_UPON_LIMIT=0 \
        CUDA_INJECTION64_PATH="$FRAMEWORK/util/tracer_nvbit/tracer_tool/tracer_tool.so" ./vecadd
    step vecadd-post "$FRAMEWORK/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing" \
        "$GATE/traces" -j8 --text
    test -s "$GATE/traces/kernelslist.g"
    step vecadd-sim "$FRAMEWORK/gpu-simulator/bin/release/accel-sim.out" \
        -trace "$GATE/traces/kernelslist.g" \
        -config "$FRAMEWORK/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3070/gpgpusim.config" \
        -config "$FRAMEWORK/gpu-simulator/configs/tested-cfgs/SM86_RTX3070/trace.config"
    grep -F 'GPGPU-Sim: *** exit detected ***' "$WORK/logs/vecadd-sim.log"
    grep 'gpu_tot_sim_cycle =' "$WORK/logs/vecadd-sim.log"
    echo "Gate artifacts: $GATE"
    ;;
*) echo "Usage: $0 {toolchain|build|gate|config}"; exit 2 ;;
esac
