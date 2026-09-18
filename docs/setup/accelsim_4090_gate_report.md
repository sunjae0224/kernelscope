# Accel-Sim 2.0.0 RTX 4090 gate report

Date: 2026-09-18. Checkout branch: `sim-track-4090`.
Artifacts: `/home/skkai/accelsim/`; no system compiler or driver changes.

## 1. Gate result

**D1 PASS; D2 functional replay PASS; D3 PARTIAL / BLOCKED; D4 deferred.**

The toolchain is usable. FA2 B1 ratios meet the 0.7–1.4 target; FlashDecoding
B1 ratios do not. B16 and naive validation need an idle GPU window, and the
attention timing model must not yet be described as calibrated.

The host's driver 580.95.05 successfully ran NVBit 1.8. The unmodified v2.0.0
simulator replayed a vector-add trace from this RTX 4090 using SM86_RTX3070:
**524,288 warp instructions, 36,589 cycles, 20.29 s wall time, clean exit**.
The application was compiled for sm_86 and executed on the physical sm_89 GPU.

Native sm_89 SASS initially produced `unsupported binary version: 89` and exit
code **0**, without the clean-exit sentinel. A documented local patch maps its
common instructions to the existing Ampere opcode map. With that patch, native
sm_89 vector-add replayed on SM86_RTX3070 at **36,859 cycles**, and on the new
SM89_RTX4090 model at **18,803 cycles**. This verifies only the exercised opcode
subset, not a complete Ada ISA implementation or calibrated timing model.

## 2. Versions and host evidence

| Component | Resolved version / evidence |
|---|---|
| Host | Ubuntu 20.04.6, glibc 2.31, 32 CPUs, 62 GiB RAM |
| GPU / driver | RTX 4090 / 580.95.05 |
| Accel-Sim | v2.0.0, `64653015f85fb5664c84a10f48527e8897d289d0` + [sm_89 patch](accelsim-sm89.patch) |
| GPGPU-Sim | `91880c53383d5a6a6742bfb1be2c5f34e39c7871` (same as prior A100 gate) |
| pybind11 | `296d5d1d3664dd0e0616e81920342e56b7249171` |
| NVBit | 1.8, from the pinned framework's `install_nvbit.sh` |
| GCC/G++ | conda-forge 11.4.0 (requested major 11, not system 9.4) |
| CMake | 3.31.8 |
| nvcc | 12.9.86, installed in `accelsim-build`; CUDA 10.1 and 13.3 not used |
| Build Python | `/home/skkai/miniforge3/envs/accelsim-build/bin/python`, 3.11.16 |
| Runtime Python | `/home/skkai/miniforge3/envs/gradkernel/bin/python`, 3.11 |
| PyTorch | 2.8.0+cu128; CUDA properties query succeeded |
| FlashAttention source | v2.8.3.post1, `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2`, clean checkout |
| FlashAttention | requested 2.8.3.post1 prebuilt wheel cannot load on glibc 2.31; same-version local build and import passed |

Full command evidence: `/home/skkai/accelsim/logs/host-versions.txt`.
Exact build package URLs: [accelsim_4090_conda_explicit.txt](accelsim_4090_conda_explicit.txt).
Both conda-forge and nvidia channels were supplied; channel priority resolved
the CUDA 12.9 components from conda-forge's NVIDIA toolkit repackaging, rather
than the task's requested nvidia-channel builds. The explicit file records this
deviation; all compiler/runtime versions above were verified from the installed tools.
Restore that environment using `conda create -p <prefix> --file <explicit-file>`.
Runtime package snapshot: `/home/skkai/accelsim/logs/gradkernel-pip-freeze.txt`.

CUDA device-query output from the native vector-add binary:

```text
GPU=NVIDIA GeForce RTX 4090 CC=8.9 SMs=128 L2=75497472 regs=65536 shared=102400 optin=101376 threads=1536
PASS: vecadd result correct (N=1048576)
```

The initial sandbox's `nvidia-smi` driver-communication failure was a sandbox
device-access restriction; the same command on the host succeeded. All GPU
commands require host GPU access in this execution environment.

## 3. Build commands and timing

Replay script: [env/setup_accelsim_4090.sh](../../env/setup_accelsim_4090.sh).

```bash
cd /home/skkai/AI_Accelerator/kernelscope
bash env/setup_accelsim_4090.sh toolchain
bash env/setup_accelsim_4090.sh build
bash env/setup_accelsim_4090.sh gate
bash env/setup_accelsim_4090.sh config
# Also verify native Ada SASS after the documented compatibility patch:
VECADD_ARCH=sm_89 bash env/setup_accelsim_4090.sh gate
```

| Step | Measured wall seconds | Result |
|---|---:|---|
| First full `cuda-toolkit` conda attempt | not instrumented | rolled back: gdb post-link script |
| Minimal CUDA 12.9 environment (cached downloads) | 9.96 | pass |
| Initial CMake configure | 1.76 / 0.13 | missing bison, then zlib development files |
| CMake configure after dependencies | 2.50 | pass |
| First simulator compile | 11.02 | missing `cudaProfiler.h` |
| Python path correction, configure | 0.26 | pass |
| Simulator incremental build | 6.27 | pass |
| Simulator install | 0.04 | pass |
| NVBit tracer, sm_89 instrumentation code | 5.03 | pass |
| Postprocessor | 1.36 | pass |
| sm_86 vector-add compile / native run | 1.51 / 0.23 | pass |
| NVBit trace / postprocess | 4.93 / 0.56 | pass |
| Unpatched RTX3070 replay | 20.29 | 36,589 cycles |
| RTX4090 replay, sm_86 input | 33.07 | 18,939 cycles |
| Patched RTX3070 replay, sm_89 input | 18.84 | 36,859 cycles |
| Patched RTX4090 replay, sm_89 input | 48.92 | 18,803 cycles; concurrent source compilation |
| gradkernel create / PyTorch install | 8.04 / 244.68 | pass |
| FlashAttention same-version local wheel | 1518.55 | import and hardware correctness checks pass |

These are per-step timings, not clean-build throughput claims: some compilation
steps reused objects after missing dependencies were fixed. Clone/download steps
started before timing instrumentation and have no claimed wall-time numbers.
All instrumented attempts are retained in `/home/skkai/accelsim/logs/timings.txt`.
The script selects absolute compilers and includes CUDA's conda
`targets/x86_64-linux/{include,lib}` paths. It compiles the postprocessor explicitly
with GCC 11 because its upstream Makefile hardcodes `g++`.

For a **new, absent** gradkernel environment, the runtime setup is:

```bash
/home/skkai/miniforge3/bin/conda create -y -p /home/skkai/miniforge3/envs/gradkernel \
  -c conda-forge python=3.11 pip
/home/skkai/miniforge3/envs/gradkernel/bin/python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
/home/skkai/miniforge3/envs/gradkernel/bin/python -m pip install \
  pandas pyarrow pytest pyyaml matplotlib streamlit plotly einops
bash docs/setup/build_flash_attn_4090.sh
/home/skkai/miniforge3/envs/gradkernel/bin/python -m pip install --no-deps \
  /home/skkai/accelsim/wheels/flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl
```

Do not recreate or downgrade a shared environment. The requested prebuilt wheel
was attempted on this machine, then replaced by a same-version local build;
future setup on Ubuntu 20.04 should take the source route directly.

## 4. Tracer gate

The sample contains exactly one vecAdd launch (4096 blocks, 256 threads).
The tracer used `DYNAMIC_KERNEL_RANGE='1@.*vecAdd.*'`, `ACTIVE_FROM_START=1`,
`NVBIT_INSTRUMENTATION_ENABLED=1`, and `TERMINATE_UPON_LIMIT=0` so result checking
could finish. The native result was checked element by element.

```text
kernel-1-ctx_0x55d6b569d760.trace.xz, _Z6vecAddPKfS0_Pfi, 4096, 1, 1, 4096, 256, 1, 1, 256, 524288,524288
```

Gate artifact directories:

- `gate-20260918-160812`: native sm_89 trace, unpatched simulator rejection.
- `gate-20260918-160856`: sm_86 trace, unpatched RTX3070 gate pass.
- `gate-20260918-161449`: native sm_89 trace, patched gate pass.

All paths above are under `/home/skkai/accelsim/`. Each has `traces/kernelslist.g`
and a nonempty `.traceg.xz`. The real sm_89 stats and full RTX4090 replay stdout
are checked into `tests/fixtures/SM89_RTX4090_*`.

## 5. Simulator and validation

Representative native sm_89 replay evidence:

```text
gpu_tot_sim_cycle = 18803
gpu_tot_sim_insn = 15728640
gpu_tot_ipc = 836.4963
L2_total_cache_accesses = 393216
L2_total_cache_misses = 131072
L2_total_cache_miss_rate = 0.3333
GPGPU-Sim: *** simulation thread exiting ***
GPGPU-Sim: *** exit detected ***
```

Log: `/home/skkai/accelsim/logs/vecadd-native89-model.log`.
Run from an artifact directory: the simulator also creates instruction and
performance-counter files in its current directory.

All seven variants also passed actual replay of that same native-sm89 vector-add
trace, not just config parsing. These are **vector-add**, not attention, results:

| variant | cycles | wall s (with concurrent compilation) |
|---|---:|---:|
| base | 18803 | 48.92 |
| l2_x2 | 18803 | 48.47 |
| l2_half | 18803 | 49.33 |
| bw_x2 | 18803 | 47.72 |
| bw_half | 18803 | 47.83 |
| sm_x2 | 18229 | 89.38 |
| sm_half | 20476 | 24.92 |

Evidence: `/home/skkai/accelsim/vecadd-variants/summaries.jsonl` and each
variant's `sim.log` and generated config in that directory.

Attention validation run: `20260918-validation2`. Hardware rows are in
`results/hw_4090_simtrack/20260918-validation2`; simulator rows are in
`results/sim_4090/20260918-validation2`. The nested `smoke/` directory is kept
separate so smoke measurements cannot affect medians in the full table.
Execution log: `/home/skkai/accelsim/logs/validation-4090-v2.log`.

| kernel | cell | real kernel_time_us | sim gpu_tot_sim_cycle | sim_us | sim/real |
|---|---|---:|---:|---:|---:|
| fa2 | B1 L1K | 63.584 | 170012 | 67.465 | **1.061** |
| flashdecoding | B1 L1K | 10.016 | 39077 | 15.507 | **1.548 — outside target** |
| fa2 | B1 L8K | 494.336 | 1300168 | 515.940 | **1.044** |
| flashdecoding | B1 L8K | 27.456 | 185450 | 73.591 | **2.680 — outside target** |
| fa2 | B16 L1K | blocked | — | — | — |
| flashdecoding | B16 L1K | blocked | — | — | — |
| naive_exec | B1 L1K, fp32 | blocked | — | — | — |

Clock convention: **sim_us = gpu_tot_sim_cycle / 2520**. The profiler's kernel
time is compared with simulated cycles; CUDA-event latency is a different
stored metric and is not used for these ratios. Correctness checks passed for
all four measured attention cells. Native naive fp32 was built but not measured.

GPU hygiene: a foreign dataset job PID 895484 delayed attention work until
16:40 KST. The four hardware cells were measured at 0% reported utilization,
with only the persistent `rerun_cli/rerun` desktop viewer (PID 861322, 440 MiB)
present. The real-HW runner records this PID as a contention warning; those
warnings are preserved rather than suppressed. A second foreign Python job,
PID 998528, subsequently occupied about 9 GiB at 67–70% utilization. At the next idle gate a replacement job PID 1017901 was active at 71%;
the script stopped with `BLOCKED: require idle GPU`. Further hardware
measurements are blocked until these jobs finish. Neither job was stopped.
The read-only clock log is `/home/skkai/accelsim/logs/validation-clocks.csv`;
observed clocks varied, so 2520 MHz is a documented nominal convention, not a
claim that the GPU was locked to that frequency.

The FA2 B1 cells launch 8 CTAs on 128 SMs (6.25% coverage); all resource
sensitivities are below 0.2%, giving the same **starved** verdict as A100.
FlashDecoding B1 L1K is **bandwidth-bound** in this model: halving bandwidth adds
40.63% cycles and doubling it removes 13.84%. These are simulator sensitivities,
not validated hardware predictions while its timing ratio is outside target.
All seven variants completed on each of the four measured attention cells
(**28/28 clean replays**). FD B1 L8K is bandwidth-bound (+88.71% for half BW,
−37.24% for double BW), unlike the prior A100 parallelism verdict. Its 256-CTA
main launch fills this GPU; a lower nominal memory bandwidth and different
cache state are plausible contributors. Doubling SMs actually increases
cycles 36.09%, while halving them reduces cycles 11.26%, suggesting model
memory contention; do not interpret these as hardware scaling predictions.
B16's expected bandwidth/parallelism verdict is still untested.

Calibration experiments used isolated config copies and the exact same source
trace. **None of these probes changed the canonical D2 configuration.**

| Trace / one changed knob | base cycles | probe cycles | probe sim/real | Interpretation |
|---|---:|---:|---:|---|
| FD B1 L1K smoke; launch latency 5000 → 0 | 39074 | 29086 | 1.152 | removes nearly 10000 cycles across two launches |
| FD B1 L1K smoke; core/icnt/L2 2520 → 2685 MHz | 39074 | 40190 | 1.494 (divide by 2685) | observed boost alone does not fix the error |
| FD B1 L1K smoke; DRAM latency 254 → 190 | 39074 | 38991 | 1.545 | little effect; no measured basis for adopting 190 |
| FD B1 L8K; launch latency 5000 → 0 | 185450 | 174091 | 2.516 | launch latency alone does not fix L8K |
| FD B1 L1K; zero launch delay + one scoped warm-up pair | 39077 | 20982 (second pair only) | **0.831** | exploratory warm-cache convention meets target |
| FD B1 L8K; zero launch delay + one scoped warm-up pair | 185450 | 72924 (second pair only) | **1.054** | exploratory warm-cache convention meets target |

Logs and configs: `/home/skkai/accelsim/calibration/{launch0,clock2685,dram_latency190,launch0_L8K}/`.
Smoke and full runs have distinct trace addresses; the 39074/39077 difference
is retained, not silently merged. The launch-delay probe follows the inherited
`-gpgpu_kernel_launch_latency 5000` and its countdown in `stream_manager.cc`;
it is relevant because a profiler kernel duration excludes host launch delay.

The scoped warm-up experiment supports cache-state mismatch as a major contributor. The three fp16 cells contain
4, 32, and 64 MiB of K/V respectively (`2 * B * L * 8 * 128 * 2` bytes), all
within the 4090's 72 MiB L2. Real-HW measurement follows warm-up. In contrast,
the observed attention `kernelslist.g` files contain only the scoped kernels,
**no `MemcpyHtoD` prefill entries**, and the B1 L1K FD base replay reports a
0.9437 L2 miss fraction. Upstream `tracer_tool.cu` records only synchronous
`API_CUDA_cuMemcpyHtoD_v2`, not the async copy callbacks. A CPU-only experiment repeated exactly the two saved target launches once,
with launch latency set to zero, then measured the second pair by subtracting
cumulative cycles after the first pair. L8K cumulative cycles were
`162796, 174091, 235719, 247015`: the measured pair is 72924 cycles (28.938 us).
L1K cumulative cycles were `24425, 29420, 45408, 50402`: the measured pair is
20982 cycles (8.326 us). Both second-pair ratios meet the target. L8K's cumulative
L2 miss fraction drops from 0.9699 after the first pair to 0.4850 after both,
consistent with a mostly warm second pair. These are explicitly **exploratory
warm-cache results**, not replacements for the baseline table or the stored
28-variant sweep. No invented prefill records or unfiltered PyTorch traces
were used. Matching this state across B16 and all what-if variants, and choosing
an explicit warm-up accounting convention, remains required before promotion.
Reproduction scripts, logs, generated configs and JSON summaries are under
`/home/skkai/accelsim/calibration/warm_launch0_{L1K,L8K}/`; each `replay.py` uses
only the already saved scoped trace. L1K took 19.06 s; L8K took 200.34 s. The L8K hardware analytic byte rate (1223 GB/s) also
exceeds the 1008 GB/s theoretical DRAM rate, consistent with cache reuse, but
this is not a hardware-counter measurement. ncu remained disabled.

CPU throughput from vector-add is 524288 / 33.07 = **15,854 warp-inst/s**
(sm_86 input), or 10,717 warp-inst/s during concurrent source compilation.
Measured attention base replay rates with four CPU workers are about
**41,984 warp-inst/s** for FA2 B1 L1K (592072 / 14.102 s), **42,206** for FA2
B1 L8K (4570312 / 108.287 s), and **29,794** for FD B1 L1K (446040 / 14.971 s).
FD B1 L8K was slower: **11,638 warp-inst/s** at base (2657824 / 228.371 s)
and **4,493** with `sm_x2` (591.566 s). The final estimator therefore uses
**10,000 warp-inst/s** for base planning, replacing the preliminary vector-add
15,000 estimate. This is not an upper bound on wall time. Use `--sim-rate 4000`
for a conservative estimate across these seven variants. Each budget estimate
is per variant; seven variants consume seven replay jobs. These workloads do
not support one universally accurate instructions/second constant.

## 6. Configuration derivation

See the [per-knob table](sm89_rtx4090_config.md). Core changes are 128 SMs,
72 MiB L2, 384-bit memory bus, nominal 2520 MHz core and 5250 MHz DRAM command
clock, 24 resident CTAs, and 99 KiB opt-in shared memory per block. Parent HMMA
modeling and `gpgpu_tensor_core_avail` stay unchanged.

The pinned parent actually has `gpgpu_n_mem=16`, buswidth 2 bytes, and
`gpgpu_shader_cta=32`; these differ from the task's anticipated 8 channels and
16 CTAs. The 4090 model uses 24 x 16-bit controllers. Its L2 geometry is a
capacity-equivalent approximation, not a measurement of physical associativity.

## 7. Problems and resolutions

1. Full CUDA toolkit conda install failed in gdb's post-link script with
   `CONDA_BACKUP_CXX: unbound variable`. Installed CUDA 12.9 components directly;
   no CUDA 13.3 fallback was needed.
2. Added absent flex, bison, zlib, OpenGL development files and CUDA profiler
   headers inside `accelsim-build`; no sudo or system packages changed.
3. Explicit Python interpreter/header/library paths resolved CMake's stale
   Python 3.12 discovery and missing `Development.Module` error.
4. Optional `~/.local/bin/stubgen` is absent; upstream permits that target to fail.
5. Native binary version 89 is not recognized by upstream v2.0.0. See the small
   local opcode-map patch; the backend now records `unsupported_binary:89`
   explicitly and propagates stderr, nonzero process exits, and timeouts.
6. Original L2 index function `P` asserted at 1024 sets. `X` is supported for
   the base and both size variants. Failed log: `vecadd-sm89-ipoly-failed.log`.
7. Requested FlashAttention wheel failed with
   `ImportError: /lib/x86_64-linux-gnu/libc.so.6: version GLIBC_2.32 not found`.
   Same-version source build (1518.55 s; import passed) uses [build_flash_attn_4090.sh](build_flash_attn_4090.sh),
   CUDA 12.9, GCC 11, sm_80 SASS, four compile workers. Existing qwen3's
   torch 2.6.0+cu124 / flash-attn 2.7.4.post1 imports, but no results have been
   substituted from that different stack.
8. Missing `kernelscope/results/store.py` was restored as explicitly authorized;
   `.gitignore` now preserves that package. Four existing store tests pass.
9. The old `lerobot3` environment has an unrelated `tests` package that shadows
   the repository namespace; its sweep tests fail import. The requested clean
   gradkernel environment runs all 69 current simulation/store tests successfully.

10. First PyTorch trace accidentally found system CUDA 10.1 `cuobjdump` through
    PATH and failed with `Value sm_89 is not defined for option gpu-architecture`.
    `tracer_env` now prepends the CUDA 12.9 build environment tools;
    `ACCELSIM_CUDA_ROOT` overrides this path. The retried FA2 smoke cell passed.

## 8. Replay and hand-off

The task requires sequential deliverable gates. While D3 is blocked, **D4's
default-root/default-architecture and README quick-start switch are deferred**.
The validation script supplies this host's paths and SM89_RTX4090 explicitly,
so it does not rely on the old A100 defaults. After calibration, set
`paths.DEFAULT_ROOT` to `/home/skkai/accelsim/accel-sim-framework`, the simsweep
work directory to `/home/skkai/accelsim/kernelscope_sim`, and architecture to
`SM89_RTX4090`; device index is already 0. Then update only README quick-start
paths/environment and finish the outstanding validation cells.

Once the GPU is idle:

```bash
cd /home/skkai/AI_Accelerator/kernelscope
bash docs/setup/validate_4090.sh
# The script prints its separate HW and simulation result directories.
/home/skkai/miniforge3/envs/gradkernel/bin/python -m kernelscope.cli report \
  --results results/hw_4090_simtrack/20260918-validation2 results/sim_4090/20260918-validation2 \
  --clock-mhz 2520 --out results/sim_4090/20260918-validation2/validation.csv
```

The validation script refuses a busy GPU, traces only scoped launches, runs
the smoke cell, the three attention cells and seven variants, then naive fp32
B1 L1K. `naive_attn` has been compiled with the conda nvcc and added sm_89
gencode. Before running, reuse a complete compatible `results/hw_4090` dataset
if another agent has produced it; none existed when checked during setup.

No edits were made to analysis, plugins, real-HW, workload, reference, checks,
`run_kernel.py`, or `pyproject.toml`. The long-format result schema and
`backend="sim:<variant>"` are unchanged. Shared report clock default is still
1410 MHz: always pass 2520 explicitly for this model.

CPU-only replay after a deliberate config revision (uses a fresh result directory):

```bash
/home/skkai/miniforge3/envs/gradkernel/bin/python -m docs.setup.replay_4090 \
  --source-results results/sim_4090/20260918-validation2 \
  --results results/sim_4090/revised-model
```

The replay helper was exercised on both saved smoke traces with `--variants base`;
both completed successfully without GPU tracing. Results are in
`results/sim_4090/20260918-replay-check`; each copied trace has provenance metadata.
CPU checks (**69 passed**):

```bash
/home/skkai/miniforge3/envs/gradkernel/bin/python -m pytest tests/test_accelsim_*.py tests/test_store.py -q
```
