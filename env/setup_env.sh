#!/usr/bin/env bash
# Exact commands used to build /scratch/uceeeee/conda_envs/gradkernel
# (GPU kernel-evaluation harness: PyTorch + FlashAttention + Nsight Compute)
#
# Machine: 4x A100-SXM4-80GB, driver 595.71.05 (CUDA 12.x/13.x runtime-capable),
# CUDA toolkit 12.9 at /usr/local/cuda.
#
# IMPORTANT: `conda activate` is broken on this machine (shell profile clobbers
# PATH). Every command below uses absolute binary paths. Do not `conda activate`.
#
# /scratch/uceeeee is slow NFS (~5 MB/s). /var/tmp is local NVMe. Pip cache and
# tmp dirs are redirected to /var/tmp for every pip call so downloads/extraction
# don't hit NFS (only the final `site-packages` write does, which is why installs
# still take tens of seconds to ~2 minutes each).
set -euo pipefail

export PIP_CACHE_DIR=/var/tmp/uceeeee/pip-cache
export TMPDIR=/var/tmp/uceeeee_tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

ENV_PREFIX=/scratch/uceeeee/conda_envs/gradkernel
CONDA_BASE=/opt/anaconda3   # from: bash -lc 'conda info --base'
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

# 1. Create the env (python 3.11.16 resolved).
"$CONDA_BASE/bin/conda" create -p "$ENV_PREFIX" python=3.11 -y

# 2. torch — CUDA build.
# Newest torch 2.x with a prebuilt flash-attn 2.x wheel (see step 3) is torch
# 2.8.0. torch 2.8.0 wheels exist for cu126/cu128/cu129 (NOT cu121/cu124) per
# https://download.pytorch.org/whl/cu12x/torch/ — cu128 chosen as the closest
# match to the installed 12.9 toolkit and the most common pairing.
"$PIP" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# Result: torch 2.8.0+cu128, triton 3.4.0 (pulled in as a torch dependency),
# torch._C._GLIBCXX_USE_CXX11_ABI == True (manylinux_2_28 wheel).

# 3. flash-attn — prebuilt wheel matching torch2.8 / cu12 / cp311 / cxx11abiTRUE.
# Wheel list verified via GitHub API against
# https://github.com/Dao-AILab/flash-attention/releases (tag v2.8.3.post1);
# flash-attn wheels only distinguish cu11 vs cu12 (not cu-minor), so the
# "cu12torch2.8" wheel is correct for our cu128 torch build.
FA_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
"$PIP" install "$FA_WHEEL_URL"
# Result: flash-attn 2.8.3.post1 (+ einops 0.8.2 dependency). No compilation —
# pure wheel install, ~15s.

# 4. flashinfer (optional). Both packages are pure-Python / prebuilt-cubin
# wheels (py3-none-any) with no torch/CUDA version pinning in the wheel name,
# BUT flashinfer's jit/env.py hard-checks that flashinfer-python and
# flashinfer-cubin are the SAME version at import time. flashinfer-cubin's
# latest published release is 0.6.13 (no 0.6.14+ cubin release exists yet),
# while a bare `pip install flashinfer-python` pulls 0.6.18 — pin
# flashinfer-python down to match, or `import flashinfer` raises
# RuntimeError("flashinfer-cubin version (0.6.13) does not match flashinfer
# version (0.6.18)"). Verify versions before pinning:
#   python3 -c "import json,urllib.request as u; [print(p, json.load(u.urlopen(f'https://pypi.org/pypi/{p}/json'))['info']['version']) for p in ('flashinfer-python','flashinfer-cubin')]"
"$PIP" install "flashinfer-python==0.6.13"
"$PIP" install flashinfer-cubin
# Result: flashinfer-python 0.6.13, flashinfer-cubin 0.6.13 (matched pair).
# `python -c "import flashinfer; print(flashinfer.__version__)"` -> "0.6.13 OK".

# 5. Analysis/report stack.
"$PIP" install pandas pyarrow pytest streamlit plotly matplotlib numpy pyyaml

# --- Verification (see docs/setup/env_report.md section 3 for full output) ---
CUDA_VISIBLE_DEVICES=0 "$PY" -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda(); print('ok')"
# flash_attn_with_kvcache num_splits=1 vs 8 decode call, SDPA backend sweep
# (CUDNN/EFFICIENT/FLASH/MATH), and CUDA-kernel-name capture via
# torch.profiler are scripted separately — see env_report.md for the exact
# snippets and results.

# --- ncu smoke test (see env_report.md section 3 — BLOCKED on this machine) ---
# CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
#   --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size \
#   --csv -k regex:flash --launch-count 1 \
#   "$PY" -c "<flash_attn_with_kvcache decode call>"
# Fails system-wide with:
#   ==ERROR== Profiling failed because a driver resource was unavailable.
#   Ensure that no other tool (like DCGM) is concurrently collecting
#   profiling data.
# Root cause: /etc/dcgm-exporter/default-counters.csv enables
# DCGM_FI_PROF_GR_ENGINE_ACTIVE / DCGM_FI_PROF_PIPE_TENSOR_ACTIVE /
# DCGM_FI_PROF_DRAM_ACTIVE / DCGM_FI_PROF_PCIE_{TX,RX}_BYTES, and the running
# dcgm-exporter.service continuously re-polls these DCP (Datacenter Profiling)
# hardware counters, which are a GPU-exclusive resource shared with Nsight
# Compute. `dcgmi profile --pause` (the documented workaround, confirmed
# effective at the DCGM level — DCGM_FI_PROF_GR_ENGINE_ACTIVE reads N/A while
# paused) is not enough because dcgm-exporter's own polling loop re-acquires
# the counters faster than a several-second python+torch cold start can hand
# ncu the window it needs. This reproduces on every GPU (0-3, busy and idle
# alike) and is independent of sandboxing. Fix requires root: stop
# dcgm-exporter.service for the duration of profiling, or remove the
# DCGM_FI_PROF_* lines from /etc/dcgm-exporter/default-counters.csv and
# restart the service. RmProfilingAdminOnly=0 confirms non-root *permission*
# to read HW counters is otherwise correctly configured on this host.
