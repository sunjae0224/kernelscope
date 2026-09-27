# Task for Codex: stand up kernelscope's Accel-Sim simulation track on this RTX 4090 host

You are setting up and validating the **simulation track only** of the `kernelscope`
project on a new machine. Another agent is concurrently working on everything else in
the same checkout. Read this whole file before running anything.

Repo: `/home/skkai/AI_Accelerator/kernelscope` (git, branch `main`, one commit).

## 1. What kernelscope is (30 seconds)

A harness that takes an *existing* GPU inference kernel (FlashAttention-2, FlashDecoding,
SDPA backends, a Triton kernel, a plain CUDA binary) as a plugin and evaluates it on two
tracks joined by one workload key:

1. **real-HW track** (Nsight-free): CUDA-event latency, `torch.profiler` kernel time and
   launch geometry, an analytic bytes/FLOPs model against measured ceilings.
2. **simulation track**: NVBit trace of the target launch(es) only, replayed in
   Accel-Sim 2.0 on a per-GPU config plus seven *what-if* variants
   (`l2_x2/l2_half`, `bw_x2/bw_half`, `sm_x2/sm_half`, `base`). Sensitivities become a
   bottleneck verdict ("bandwidth-bound", "parallelism-bound", "starved", "insensitive").

Both tracks write long-format parquet rows `(workload_key, kernel, backend, metric, unit,
value, launch_idx, ...)` through `kernelscope.results.store.ResultStore`; simulation rows use
`backend="sim:<variant>"`. Downstream code (`kernelscope/analysis/report.py`, and a new
surrogate performance model the other agent is building) consumes those rows. **That
contract must not change.**

Read, in this order:
- `README.md`
- `docs/STATUS.md` (what worked on the previous A100 host, with real numbers)
- `docs/plan/2026-09-01-project-plan.md` section 2.4 (simulation backend design, opcode
  coverage, tracing rules, budget table)
- `docs/setup/accelsim_gate_report.md` (the A100 gate report; replicate its structure and
  its build steps, adapted to this host)
- `kernelscope/backends/accelsim/*.py`, `tests/test_accelsim_*.py`, `tests/fixtures/`
- `kernelscope/run_kernel.py` `--mode trace` and `kernelscope/backends/accelsim/trace.py`
  (instrumentation-window mechanism: tracer starts with
  `NVBIT_INSTRUMENTATION_ENABLED=0` and the run enables it around one launch)
- `grids/sim_smoke.yaml`, `grids/w1_min_B1.yaml`, `grids/w1_min_B16.yaml`

## 2. Situation and ownership

- The previous host (`geneva.ee.ucl.ac.uk`, 4x A100, where every number in `STATUS.md`
  was produced) is **no longer accessible**. Nothing from `/scratch/uceeeee` or
  `/var/tmp/uceeeee` exists here. All paths in the repo that point there are stale.
- **You own**: the Accel-Sim/NVBit toolchain build, a new `SM89_RTX4090` simulator
  config, the simulation backend defaults for this host, sim-vs-real validation, and the
  docs for all of that.
- **You may edit**: `kernelscope/backends/accelsim/*`, the `simsweep` subcommand block in
  `kernelscope/cli.py`, `tests/test_accelsim_*.py` and new fixtures under `tests/fixtures/`,
  `docs/setup/*`, `docs/STATUS.md` (append only, under your own dated heading),
  `env/setup_accelsim_4090.sh` (new), `grids/sim_*.yaml`, `samples/naive_attn/Makefile`
  (only to add an sm_89 gencode), `.gitignore` (only for simulation artifacts).
- **Do not edit** (owned by the other agent): `kernelscope/analytic.py`,
  `kernelscope/analysis/`, `kernelscope/plugins/`, `kernelscope/backends/realhw/`,
  `kernelscope/bench/`, `kernelscope/workload.py`, `kernelscope/reference.py`,
  `kernelscope/check.py`, `kernelscope/run_kernel.py` (if the trace mode truly needs a
  change, make the smallest one and record it in your STATUS entry), `pyproject.toml`,
  `README.md` beyond the quick-start paths for this host.
- Work on a branch `sim-track-4090`, small commits per deliverable, never commit to
  `main`. Check `git status` / `git log --all` before touching a shared file; the other
  agent may have landed it first.
- Known repo defect: `kernelscope/results/store.py` is **missing from the clone** because
  `.gitignore`'s `results/` pattern swallowed the package subdirectory, so
  `kernelscope.cli` fails to import. If it is still missing when you need it, write a
  minimal implementation that satisfies `tests/test_store.py` (parquet, one file per
  `write`, `load` concatenates, `extra` columns stamped on every row, empty store returns
  an empty frame with the schema) and add `!kernelscope/results/` to `.gitignore`. The
  other agent may do this first; if the file exists, leave it alone.

## 3. Machine facts (verified 2026-09-18; re-verify anything you rely on)

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (AD102), compute capability 8.9 |
| SMs | 128 (1 SM per cluster in Accel-Sim terms) |
| L2 | 75,497,472 B = 72 MiB (full AD102 has 96 MiB; 4090 has 72 MiB enabled) |
| memory | 23.5 GiB GDDR6X, 384-bit bus, memory clock 10,501 MHz (21 Gbps effective, ~1008 GB/s theoretical) |
| per-SM limits | 1536 threads, 65,536 registers, 100 KB shared memory (99 KB opt-in per block), warp 32 |
| clocks | boost 2520 MHz (torch), nvidia-smi max SM clock 3105 MHz; decide and document which clock the config uses and which you use to convert cycles to seconds |
| PCIe | gen 4 |
| driver | 580.95.05 (CUDA 13.0-capable). NVBit 1.8's README says driver <= 575.xx; on the A100 host it worked on 595.71 anyway. Treat as a gate, not a blocker. |
| OS / system tools | Ubuntu 20.04.6, gcc 9.4.0, cmake 3.16.3. **Do not build Accel-Sim with these**: Accel-Sim issue #380 reports gcc 9.4 build failures and the A100 host used gcc 11.5 / cmake 3.31. |
| CUDA toolkits | `/usr/bin/nvcc` is **CUDA 10.1** from apt: never use it. Real toolkit: `/usr/local/cuda-13.3` (nvcc 13.3, ncu 2026.2.1). There is **no CUDA 12.x** toolkit on the system. |
| CPU / RAM / disk | 32 cores, 62 GB RAM, 1.5 TB free on `/` (local disk, no NFS). Put builds, traces, sim logs under `/home/skkai/accelsim/`. |
| network | github.com and download.pytorch.org reachable |
| profiling counters | `RmProfilingAdminOnly=1`, so `ncu` hardware counters need root. `sudo` needs a password (the user can type it). No DCGM on this box. ncu is *opt-in* in kernelscope and not your track; but if the user enables `NVreg_RestrictProfilingToAdminUsers=0`, use `ncu --metrics lts__t_sector_hit_rate.pct,dram__bytes.sum,gpu__time_duration.sum` as an extra validation source for L2 miss rate and DRAM bytes. |
| conda | miniforge `conda 26.1.1` at `/home/skkai/miniforge3`. Always call interpreters by absolute path (`/home/skkai/miniforge3/envs/<env>/bin/python`); do not rely on `conda activate`. |
| existing envs | `qwen3`, `starc`, `rpc` each have `torch 2.6.0+cu124` + `flash_attn 2.7.4.post1` and may serve as a **stopgap** tracing env (no pandas/pytest in them). `stity` has `flashinfer 0.5.3`. `lerobot3` has pandas/pyarrow/pytest but no flash-attn. No env has both. |
| target env | `/home/skkai/miniforge3/envs/gradkernel`. **Create it if absent** following `env/setup_env.sh` adapted to this host: python 3.11, `torch==2.8.0` from the cu128 index, the prebuilt `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311` wheel (sm_80 SASS runs on sm_89), then `pandas pyarrow pytest pyyaml matplotlib streamlit plotly`. The other agent shares this env: only add packages, never remove or downgrade. |

## 4. Deliverables, in order, each gated on the previous one

### D1. Toolchain gate (replicate `docs/setup/accelsim_gate_report.md` on this host)

1. Build tools in a dedicated conda env `accelsim-build` (do not touch system gcc):
   `gxx_linux-64=11`, `gcc_linux-64=11`, `cmake>=3.22`, `boost`, `zstd`, `libxml2`,
   `openssl`, `python=3.11`, plus a **CUDA 12.9 (or 12.8) toolkit** from the `nvidia`
   channel (`cuda-toolkit` / `cuda-nvcc` / `cuda-cudart-dev`). Rationale: matches the
   known-good A100 stack; NVBit 1.8 + Accel-Sim 2.0 on CUDA 13.3 is unverified. If the
   conda CUDA 12.x route fails, fall back to `/usr/local/cuda-13.3` and document why.
   Record exactly what resolved (`conda list --explicit`).
2. Clone `accel-sim-framework` tag `v2.0.0` (commit `64653015f85fb5664c84a10f48527e8897d289d0`)
   under `/home/skkai/accelsim/`. Build `gpu-simulator` (cmake, release) and
   `util/tracer_nvbit` (`install_nvbit.sh` pulls NVBit 1.8). Watch for issues #380
   (gcc), #413 (`libcudart.so.12` path), #360 (PyTorch trace silently failing).
3. **Gate A, tracer under driver 580**: trace a vector-add sample; confirm
   `kernelslist.g` and a `.traceg`/`.tracez` with a nonzero instruction count.
4. **Gate B, simulator**: replay it on `tested-cfgs/SM86_RTX3070` (closest existing
   consumer Ampere-family config; confirm the directory exists in v2.0.0 with
   `ls gpu-simulator/gpgpu-sim/configs/tested-cfgs/`) to a clean
   `GPGPU-Sim: *** exit detected ***`; capture `gpu_tot_sim_cycle`.
5. Write a replayable `env/setup_accelsim_4090.sh` and record wall time per step.
6. If Gate A fails: try the NVBit version the tracer's `install_nvbit.sh` allows you to
   pin, check NVBit's issue tracker for driver-580 reports, and if nothing works **stop
   and report BLOCKED with the exact error**. The project's fallback is a real-HW-only
   plan, which is the user's decision, not yours.

### D2. `SM89_RTX4090` config

1. Create `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config` and
   the matching `gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config`, plus
   whatever else an arch directory needs (compare how `SM86_RTX3070` and `SM80_A100` are
   laid out, including any `config_*.icnt` file).
2. Start from `SM86_RTX3070` (same 8.x SM family: 4 sub-partitions, 128 FP32 lanes,
   1536 threads/SM, 100 KB shared memory). Change and justify **each** knob from the
   table in section 3, `nvidia-smi`, the NVIDIA Ada whitepaper, and the CUDA occupancy
   docs. At minimum:
   - `-gpgpu_n_clusters 128`, cores per cluster 1
   - clock domains `core:icnt:l2:dram`: core 2520 (or the value you justify); derive
     the DRAM entry by seeing how the 3070 config encodes GDDR6 14 Gbps and scaling to
     GDDR6X 21 Gbps
   - memory partitions / channels: 384-bit bus = 12 x 32-bit channels (check the 3070
     uses 8 for its 256-bit bus and how `-gpgpu_n_mem` and `-gpgpu_n_sub_partition_per_mchannel` relate)
   - L2: edit `-gpgpu_cache:dl2` sets/assoc/line so the total across all sub-partitions
     is 72 MiB
   - registers 65536, shared-memory config 100 KB, max CTAs per SM (Ada allows 24; GA10x
     16; verify against the CUDA programming guide table)
   - keep the 3070's tensor-core (HMMA) modelling; **do not touch**
     `gpgpu_tensor_core_avail` (see Accel-Sim issue #451)
3. Confirm `kernelscope/backends/accelsim/config.py` `VARIANTS` still parse and scale
   the new config (the `dl2` set count, the DRAM clock field, `n_clusters`). Its
   docstring assumes A100's 40 channels x 4 sub-partitions; make the comment generic.
   Add a CPU unit test that derives all 7 variants from the new config and checks each
   changed exactly one knob by the intended factor.
4. Write `docs/setup/sm89_rtx4090_config.md` with one row per knob:
   `knob | SM86_RTX3070 value | SM89_RTX4090 value | source | confidence (high/med/low)`.

### D3. Calibration and validation against real hardware

1. Cells, in this order: `grids/sim_smoke.yaml`; then `fa2` and `flashdecoding` on decode
   `B1 L1K`, `B1 L8K`, `B16 L1K` (the same cells as the A100 table in `STATUS.md`, so
   sim/real ratios are comparable across GPUs); then `naive_exec` on `B1 L1K` (build it
   with `make` in `samples/naive_attn/` using the conda nvcc; add `-gencode
   arch=compute_89,code=sm_89` if the Makefile lacks it). Do **not** add new plugins.
2. Real-HW reference: `kernelscope sweep` on the same cells with the `gradkernel` env
   (`profile` track, `kernel_time_us`). Reuse `results/hw_4090/` if the other agent has
   already produced it; otherwise write yours to `results/hw_4090_simtrack/`.
3. Produce the validation table in your report:
   `kernel | cell | real kernel_time_us | sim gpu_tot_sim_cycle | sim_us | sim/real`.
   State the clock you divided by. Target: `sim/real` in 0.7–1.4 for `fa2` and
   `flashdecoding`. If the ratio is systematically off, revisit D2 knobs (clock domains
   and the memory model first) and keep a log of which knob moved the ratio by how much.
4. Simulation budget: measure this CPU's warp-instructions/second and update the
   estimator behind `simsweep --max-sim-s` in `sweep.py` if it hard-codes the A100
   host's 27.5K figure. Put the measured number in the report.
5. Run all 7 variants on the `fa2` / `flashdecoding` cells and run
   `kernelscope report --results results/hw_4090* results/sim_4090`. Verdicts should
   qualitatively match the A100 ones in `STATUS.md` (FA2 B1 starved; FlashDecoding B16
   bandwidth-bound). Differences are informative, not failures: explain them in terms
   of the 4090's 128 SMs and lower HBM-class bandwidth.

### D4. Backend defaults for this host

- `kernelscope/backends/accelsim/paths.py`: `DEFAULT_ROOT` ->
  `/home/skkai/accelsim/accel-sim-framework` (keep the `ACCELSIM_ROOT` override).
- `kernelscope/cli.py` `simsweep`: `--work-dir` default `/home/skkai/accelsim/kernelscope_sim`,
  `--arch` default `SM89_RTX4090`, `--device-index` default 0.
- `tests/test_accelsim_*.py` must stay green on CPU; add an SM89 fixture (a stats file and
  a stdout log from a real 4090 run) alongside the existing `SM80_A100` fixtures.
- Update the README quick-start block for this host (paths and env only).

### D5. Report and hand-off

- `docs/setup/accelsim_4090_gate_report.md` with the same sections as the A100 report:
  gate result, versions/commits, build commands and timing, config derivation summary
  (link the knob table), validation table, issues hit and how they were resolved,
  replay script path.
- Append a dated entry to `docs/STATUS.md` under the heading
  `## <date> — sim track (RTX 4090)`. Do not rewrite other entries.
- Final message: what passed, what is outside target, exact commands to reproduce the
  validation table, and anything the other agent must know (for example a changed
  clock convention or a metric name that differs from the A100 stats).

## 5. Rules

- Never trace or simulate an unfiltered PyTorch process. Always scope to the target
  launch(es) with the kernel regex / `DYNAMIC_KERNEL_RANGE` mechanism already in
  `trace.py`; an unfiltered trace of a torch process is gigabytes and useless.
- Measure on an idle GPU; check `nvidia-smi` before real-HW reference runs (the sweep
  records utilisation and foreign PIDs and warns).
- Verify every claim with a command and paste the evidence in the report. If evidence
  contradicts a fact in this file, say so explicitly and go with the evidence.
- Stop and ask the user only for: a sudo password (for example to enable profiling
  counters), or a BLOCKED gate. Everything else, decide and document.
- Keep the results contract (section 1) and stay out of the other agent's files.
