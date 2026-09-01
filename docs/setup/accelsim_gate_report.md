# Accel-Sim 2.0.0 W1 gate report

Machine: 4x A100-SXM4-80GB, driver 595.71.05, CUDA toolkit 12.9 at `/usr/local/cuda`, gcc/g++ 11.5.0, cmake 3.31.8, 128 cores / 502 GB RAM. Non-root, no sudo. All cloning/building/tracing done under `/var/tmp/uceeeee/accelsim/` (local NVMe) per instructions, not on the slow `/scratch` NFS mount.

## 1. GATE RESULT: PASS

NVBit 1.8's tracer (`tracer_tool.so`) loaded and traced a real CUDA kernel cleanly under driver 595.71.05, even though NVBit 1.8's own README states `CUDA driver version: <= 575.xx`. No driver/version error was printed, no crash, no fallback needed. `post-traces-processing` converted the raw trace into simulator-consumable `kernelslist.g` + `.traceg.xz`, and `accel-sim.out` replayed it against the `SM80_A100` config to completion (exit code 0, clean `GPGPU-Sim: *** exit detected ***`), producing `gpu_tot_sim_cycle = 11333`, `gpu_sim_insn = 15728640`, full L2/L1 stat dumps, and correct vecadd output. Both halves of the gate (tracer produces a trace; simulator replays it with SM80_A100) hold.

## 2. Versions used

| Component | Version / commit |
|---|---|
| accel-sim-framework | tag `v2.0.0` (existed as documented — no fallback to `2.0.0` or `dev` needed), commit `64653015f85fb5664c84a10f48527e8897d289d0` |
| GPGPU-Sim (auto-cloned dependency) | `dev` branch, commit `91880c53383d5a6a6742bfb1be2c5f34e39c7871`, reports itself as "GPGPU-Sim version 4.2.0" |
| NVBit | v1.8 (`nvbit-Linux-x86_64-1.8.tar.bz2`, the only version `install_nvbit.sh` pulls) |
| gcc / g++ | 11.5.0 (Red Hat 11.5.0-14) |
| CUDA toolkit | 12.9.86 (`nvcc`, `/usr/local/cuda`) |
| Driver | 595.71.05 |
| Python (for pip installs / helper scripts) | `/usr/bin/python3.11` (system interpreter, **not** the conda base env that's on `$PATH` by default — see §7) |

## 3. Build commands and timing

Full replayable log: **`/var/tmp/uceeeee/accelsim/setup_accelsim.sh`**. Summary of wall-clock time per step:

| Step | Command(s) | Wall time |
|---|---|---|
| Clone accel-sim-framework | `git clone --branch v2.0.0 --depth 1 ...` | a few seconds |
| Install Python deps | `/usr/bin/python3.11 -m pip install --user -r requirements.txt` | 37s |
| `setup_environment.sh` (clones GPGPU-Sim + pybind11) | `source ./gpu-simulator/setup_environment.sh` | a few seconds |
| CMake configure | `cmake -S ./gpu-simulator/ -B ./gpu-simulator/build` | 8.8s |
| CMake build | `cmake --build ./gpu-simulator/build -j32` | 13.6s wall (2m42s CPU across 32 threads) |
| CMake install | `cmake --install ./gpu-simulator/build` | <1s |
| NVBit download | `bash util/tracer_nvbit/install_nvbit.sh` | <1s (917 KB download) |
| Tracer build | `make -C ./util/tracer_nvbit/` | 2m36s |
| vecadd compile | `nvcc -arch=sm_80 -o vecadd vecadd.cu` | a few seconds |
| Tracer run (the gate) | see §4 | <1s |
| Post-processing | `post-traces-processing ./traces -j 8 --text` | <1s |
| Simulator run on SM80_A100 | `accel-sim.out -trace ... -config ...` | **29.9s** (real) |

Disk footprint: `accel-sim-framework` tree (incl. GPGPU-Sim submodule + NVBit release) = 132 MB; smoke-test traces directory = 4.4 MB. `/var/tmp` had 28 GB free after everything (started at 29 GB free).

## 4. Tracer run — the gate

Vector-add smoke test at `/var/tmp/uceeeee/accelsim/smoke/vecadd.cu`: `1<<20` (1,048,576) floats, 256 threads/block, 4096 blocks. Compiled with `nvcc -arch=sm_80`. Plain run (no tracer) printed `PASS: vecadd result correct (N=1048576)`.

Traced invocation:
```bash
CUDA_VISIBLE_DEVICES=0 \
  DYNAMIC_KERNEL_RANGE="1" \
  ACTIVE_FROM_START=1 \
  TERMINATE_UPON_LIMIT=1 \
  CUDA_INJECTION64_PATH=/var/tmp/uceeeee/accelsim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so \
  ./vecadd
```
Output (excerpt, full env-var banner omitted):
```
------------- NVBit (NVidia Binary Instrumentation Tool v1.8) Loaded --------------
...
Writing results to /var/tmp/uceeeee/accelsim/smoke/traces//kernel-1-ctx_0x1f13110.trace.xz
PASS: vecadd result correct (N=1048576)
```
Exit code 0. No error text — nothing to report as a failure/fallback.

Files produced under `traces/` (written relative to the CWD, i.e. `/var/tmp/uceeeee/accelsim/smoke/traces/`):

| File | Size | Purpose |
|---|---|---|
| `kernelslist_ctx_0x1f13110` | 108 B | raw kernel launch list (memcopies + kernel refs) for this CUDA context |
| `stats_ctx_0x1f13110` | 255 B | per-kernel metadata: name, grid/block dims, instruction counts |
| `kernel-1-ctx_0x1f13110.trace.xz` | 2.8 MB | raw xz-compressed SASS trace for kernel 1 |

`stats_ctx_0x1f13110` content:
```
kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, #blocks, block_dimX, block_dimY, block_dimZ, #threads, total_insts, total_reported_insts
kernel-1-ctx_0x1f13110.trace.xz, _Z6vecAddPKfS0_Pfi, 4096, 1, 1, 4096, 256, 1, 1, 256, 524288,524288
```

Post-processing (converts to the simulator-consumable format):
```bash
/var/tmp/uceeeee/accelsim/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing \
  ./traces -j 8 --text
```
`--text` was used to match the README's example `-trace .../kernelslist.g` invocation (plain-text `.traceg`); the default without `--text` is compressed `.tracez`. This added two files to `traces/`:

| File | Size | Purpose |
|---|---|---|
| `kernelslist.g` | 109 B | simulator's trace-list entry point (memcopies + kernel trace filename) |
| `kernel-1-ctx_0x1f13110.traceg.xz` | 112 KB | post-processed, simulator-readable per-kernel trace (still xz-compressed on disk; the simulator decompresses transparently) |

## 5. Simulator run

```bash
/var/tmp/uceeeee/accelsim/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out \
  -trace /var/tmp/uceeeee/accelsim/smoke/traces/kernelslist.g \
  -config /var/tmp/uceeeee/accelsim/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config \
  -config /var/tmp/uceeeee/accelsim/accel-sim-framework/gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config
```
Full stdout saved to `/var/tmp/uceeeee/accelsim/smoke/sim_a100_stdout.log` (20,795 lines). Wall-clock: **29.9s** (`real 0m29.892s`, exit code 0).

Key stats (grepped from stdout):

| Stat | Value |
|---|---|
| `gpu_sim_cycle` | 11333 |
| `gpu_tot_sim_cycle` | 11333 |
| `gpu_sim_insn` | 15,728,640 |
| `gpu_tot_sim_insn` | 15,728,640 |
| `gpu_ipc` / `gpu_tot_ipc` | 1387.8619 |
| `gpu_occupancy` | 91.4766% |
| `L2_total_cache_accesses` | 393,216 |
| `L2_total_cache_misses` | 131,072 |
| `L2_total_cache_miss_rate` | 0.3333 |
| `L2_total_cache_reservation_fails` | 0 |
| `gpgpu_simulation_time` | 29 sec |
| `gpgpu_simulation_rate` | 542,366 inst/sec, 390 cycle/sec |
| `gpgpu_silicon_slowdown` | 3,615,384x |

L1D cache banks show `Miss_rate = 1.000` and nonzero `Reservation_fails` — expected for this workload (vecadd is a pure streaming/cold-cache access pattern with no reuse; not an error condition, no assertion/crash accompanied it).

## 6. A100 config what-if option values

From `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config` (and `gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config` for the trace-side settings, not separately tabulated — it only carries trace-replay knobs, none of the requested options live there):

| Option | Value |
|---|---|
| `gpgpu_n_clusters` | 108 |
| `gpgpu_n_cores_per_cluster` | 1 |
| `gpgpu_n_mem` | 40 |
| `gpgpu_clock_domains` | `1410:1410:1410:1512` (Core:Interconnect:L2:DRAM, MHz) |
| `gpgpu_shmem_size` | 167936 |
| `gpgpu_cache:dl2` | `S:128:128:16,L:B:m:L:X,A:192:4,32:0,32` |
| `gpgpu_l2_rop_latency` | 200 |
| `dram_latency` | 190 |
| `gpgpu_dram_timing_opt` | `nbk=16:CCD=1:RRD=7:RCD=22:RAS=50:RP=22:RC=72:CL=22:WL=4:CDLR=5:WR=19:nbkgrp=4:CCDL=4:RTPL=7` |

## 7. Problems hit and how they were resolved

1. **CMake configure error at `CMakeLists.txt:19`**: `if($ENV{ACCELSIM_CONFIG} STREQUAL "debug")` fails with `CMake Error ... Unknown arguments specified` when `$ACCELSIM_CONFIG` is unset. Root cause: an unset `$ENV{VAR}` reference expands to **zero tokens** (not an empty-string token) when unquoted, so the `if()` call sees only `STREQUAL "debug"` and can't parse it. Fix: always `export ACCELSIM_CONFIG=release` before running cmake (the value doesn't matter beyond avoiding "debug"; it also determines the install subdirectory `bin/$ENV{ACCELSIM_CONFIG}/` used later, so it must stay consistent between configure/build/install).
2. **Environment variables set via `source ./gpu-simulator/setup_environment.sh` didn't propagate to a later, separate shell invocation.** `$GPGPUSIM_ROOT`/`$ACCELSIM_ROOT` (which the top-level `CMakeLists.txt` reads via `$ENV{...}` at `add_subdirectory($ENV{GPGPUSIM_ROOT})`) only exist in the shell that sourced the script. Fix: source `setup_environment.sh` and run the `cmake` commands in the **same** shell invocation (this is baked into `setup_accelsim.sh`).
3. **System `python3` on `$PATH` resolves to `/opt/anaconda3/bin/python3`** (conda `base` env, auto-activated by shell rc — not something this session ran `conda activate` for, but present anyway). Per the "don't use conda activate" instruction, Python dependencies (`pyyaml`, `pandas`, `plotly`, `kaleido`, `psutil`, `tqdm`) were installed with `/usr/bin/python3.11 -m pip install --user -r requirements.txt` instead, sidestepping conda entirely. This produced a harmless `pip` dependency-resolver warning about a pre-existing system `scipy` wanting an older `numpy` than the one just installed — irrelevant to Accel-Sim, no package used by the build/trace/sim path depends on that `scipy`.
4. **`gen_pyi` custom target failure during build** (`/bin/sh: line 1: /home/uceeeee/.local/bin/stubgen: No such file or directory`): expected and harmless — the target is defined with `|| (exit 0)` specifically to allow this optional `.pyi` stub-generation step to fail without breaking the build. No `mypy`/`stubgen` was installed and none was needed.
5. Nothing required the "retry with sandbox disabled" fallback — `git clone`, `wget` (NVBit download), and `pip install` all succeeded on the first attempt from this environment.
6. No NVBit fallback was needed (see §1) — the ~20-minute fallback budget for hunting a newer NVBit release was not spent.

## 8. Notes for the harness author (programmatic invocation)

**Trace any CUDA binary:**
```bash
CUDA_VISIBLE_DEVICES=<n> \
  DYNAMIC_KERNEL_RANGE="<spec>" \
  ACTIVE_FROM_START=1 \
  TERMINATE_UPON_LIMIT=1 \
  CUDA_INJECTION64_PATH=<repo>/util/tracer_nvbit/tracer_tool/tracer_tool.so \
  <your_binary> [args...]
```
- `DYNAMIC_KERNEL_RANGE` formats: `"3"` (single), `"5-8"` (range), `"10-"` (open-ended), `"2 5-8 10-"` (space-separated multiple), `"5-8@regex1,regex2"` (range + name filter). Unset/empty traces every kernel.
- `ACTIVE_FROM_START=0` instead waits for `cuProfilerStart`/`cuProfilerStop` and makes `DYNAMIC_KERNEL_RANGE` a no-op.
- Default output directory is `./traces/` relative to CWD (i.e. `TRACES_FOLDER` defaults to `./traces`); override the destination with `USER_DEFINED_FOLDERS=1 TRACES_FOLDER=<dir>` — verified working, the tool then writes to `<dir>/traces/...` (it appends its own `traces/` subfolder, so pre-create `<dir>` not `<dir>/traces`).
- Output per traced kernel: `kernel-<id>-ctx_<hexctx>.trace.xz` plus one shared `kernelslist_ctx_<hexctx>` and `stats_ctx_<hexctx>` per CUDA context. `stats_ctx_*` is a CSV header + one row per kernel with mangled name, grid/block dims, instruction counts — useful for a harness to sanity-check it traced the intended kernel before running the simulator.
- Success is signaled by `Writing results to <path>/kernel-N-ctx_....trace.xz` in stdout and exit code 0 (with no distinct "gate passed" flag beyond that); a driver-incompatibility failure would show up as a CUDA error from the injected library or a nonzero exit / crash before that line — neither occurred here at driver 595.71.05.

**Post-process (required before simulating):**
```bash
<repo>/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing <traces_dir> -j <N> [--text]
```
- `<traces_dir>` accepts either a single `kernelslist*` file or a directory (it scans for files whose name contains `kernelslist` but not `.g`, i.e. it skips its own prior output on reruns — safe to invoke repeatedly on the same directory).
- Without `--text`: writes compressed `.tracez` output and a `kernelslist.g` unchanged in format. With `--text`: writes plain-text `.traceg.xz` (still xz'd on disk, decompressed by the simulator at load) and a `kernelslist.g` that lists memcopies plus the traced-kernel filename — this is what `-trace` should point at.
- `kernelslist.g` is written into the **same directory** as the input `kernelslist_ctx_*` file(s); a harness should locate it there rather than assuming a fixed name/location upfront.

**Simulate:**
```bash
<repo>/gpu-simulator/bin/release/accel-sim.out \
  -trace <traces_dir>/kernelslist.g \
  -config <repo>/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config \
  -config <repo>/gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config
```
- Both `-config` flags are required (first supplies GPGPU-Sim's timing/architecture model, second supplies trace-replay-specific knobs); order matters — later `-config` values can override earlier ones for keys present in both, so keep `gpgpusim.config` first and `trace.config` second as shown, matching the README.
- All stats are printed to **stdout** as `key = value` (or `key=value`, inconsistently) lines at the end of the run — no separate stats file is written by default. A harness should capture stdout and grep it; key lines: `gpu_sim_cycle`, `gpu_tot_sim_cycle`, `gpu_sim_insn`, `gpu_tot_sim_insn`, `gpu_ipc`, `gpu_tot_ipc`, `gpu_occupancy`, `L2_total_cache_accesses`, `L2_total_cache_misses`, `L2_total_cache_miss_rate`, `gpgpu_simulation_time`, `gpgpu_simulation_rate`. Clean completion is marked by the literal lines `GPGPU-Sim: *** simulation thread exiting ***` / `GPGPU-Sim: *** exit detected ***` near the end of stdout plus exit code 0 — a harness can use those as a sentinel that stats above them are trustworthy (vs. a crash mid-run that leaves partial output).
- Other configs available at the same tree depth (`gpu-simulator/configs/tested-cfgs/<CFG>/` and `gpu-simulator/gpgpu-sim/configs/tested-cfgs/<CFG>/`) include `QV100-SASS`-style names per `util/job_launching/configs/define-standard-cfgs.yml`; `SM80_A100` was used here per the assignment.
- Both the build (`gpu-simulator/build/`) and the installed binary (`gpu-simulator/bin/release/accel-sim.out`) are single-machine, non-relocated paths under `/var/tmp/uceeeee/accelsim/accel-sim-framework/` in this setup — a harness on another host must rebuild there or adjust paths; nothing was installed system-wide.
