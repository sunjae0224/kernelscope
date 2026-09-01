# kernelscope

Dual-track evaluation harness for GPU inference kernels: plug in an **existing**
kernel (FlashAttention-2, FlashDecoding, SDPA backends, flashinfer, Mamba scan,
your own CUDA/Triton kernel) and get

1. **real-hardware** numbers on an A100 — **without Nsight**: CUDA-event latency,
   torch.profiler kernel time + launch geometry (grid/block/registers/smem →
   occupancy estimate), and an analytic traffic/FLOP model that turns kernel time
   into achieved GB/s and TFLOPS against *measured* ceilings; and
2. **simulated** numbers from Accel-Sim's `SM80_A100` model plus *what-if*
   variants (L2 size, HBM bandwidth, SM count) that no measurement can give, with
   the instruction mix read straight from the NVBit trace,

joined on one workload key so the two tracks validate each other
(sim cycles vs measured kernel time) and the what-if sensitivity gets a verdict
("bandwidth-bound", "parallelism-bound", "insensitive").

Why Nsight-free? Shared GPU boxes often run `dcgm-exporter`, which holds the
hardware-counter session and makes `ncu` fail on every kernel. Everything above
uses the CUPTI *activity* API (kernel timing), NVBit instrumentation, or
arithmetic — none of which need that lock. `ncu` remains an opt-in backend
(`sweep --ncu /path/to/ncu`) for hosts where it works.

Why not just benchmark inside vLLM? vLLM answers *how fast is it here*. This
answers *why* (isolated, controlled measurements — no scheduler/CUDA-graph
confounding) and *what would change it* (counterfactual hardware). See
[docs/plan/2026-09-01-project-plan.md](docs/plan/2026-09-01-project-plan.md) §0.

## Quick start

```bash
PY=/scratch/uceeeee/conda_envs/gradkernel/bin/python      # always absolute paths on this box
$PY -m kernelscope.cli list                                # registered plugins
$PY -m kernelscope.cli ceilings --out results/machine_ceilings.json           # HBM / fp16-GEMM peaks
CUDA_VISIBLE_DEVICES=1 $PY -m kernelscope.cli sweep --grid grids/w1_min.yaml \
    --plugins fa2,flashdecoding,sdpa_flash,sdpa_efficient --results results/hw \
    --ceilings results/machine_ceilings.json --python $PY
CUDA_VISIBLE_DEVICES=2 $PY -m kernelscope.cli simsweep --grid grids/sim_smoke.yaml \
    --plugins fa2,flashdecoding --results results/sim --variants base,bw_x2,bw_half,l2_x2,l2_half,sm_x2,sm_half \
    --work-dir /var/tmp/uceeeee/kernelscope_sim --device-index 2 --python $PY
$PY -m kernelscope.cli report --results results/hw results/sim --out results/summary.csv
$PY -m kernelscope.cli plot --results results/hw --ceilings results/machine_ceilings.json --out results/roofline.png
```

Measure on an **idle** GPU: check `nvidia-smi` first (the sweep records utilisation and
foreign processes on the target GPU at start and warns). Long-running jobs of your own on
another GPU are fine; on the same GPU they inflate kernel times several-fold.

`simsweep` needs the built Accel-Sim tree (default `/var/tmp/uceeeee/accelsim/accel-sim-framework`,
override with `--accelsim-root` or `ACCELSIM_ROOT`); see [docs/setup/accelsim_gate_report.md](docs/setup/accelsim_gate_report.md).

## Layout

```
kernelscope/
  workload.py            logical workload (phase, B, L_q, L_kv, H_q, H_kv, d, dtype, causal) + grid expansion
  reference.py           pure-torch reference attention (GQA, bottom-right causal)
  check.py               plugin-vs-reference correctness (reported, never raised)
  analytic.py            compulsory bytes / FLOPs / arithmetic intensity per workload
  run_kernel.py          per-cell subprocess: --mode check | latency | profile | kernels | ncu
  plugins/
    base.py              KernelPlugin (python-callable) / ExecutablePlugin (CUDA binary)
    builtin/             sdpa_{math,efficient,cudnn,flash}, fa2, flashdecoding
  backends/
    realhw/              latency.py (CUDA events), kprofile.py (torch.profiler + occupancy),
                         ncu.py (optional), sweep.py (orchestrator)
    accelsim/            paths / config (what-if variants) / trace / stats / sweep
  analysis/
    trace_mix.py         opcode histogram + global bytes from the raw NVBit trace
    report.py            one wide row per (kernel, workload) across tracks + what-if verdict
  bench/ceilings.py      measured HBM and fp16-GEMM peaks (torch only)
  results/store.py       long-format parquet, one file per write
grids/                   workload grids (w1_min, decode_full, prefill, sim_smoke)
tests/                   CPU unit tests + gpu-marked integration tests
docs/plan/               approved plan;  docs/setup/  machine reports;  docs/STATUS.md  progress log
```

## Registering your own kernel

**Python-callable kernel** (torch extension, Triton, library op) — one file. Implement
`build_inputs(workload)` in *your* layout (dense, paged, whatever), `run(inputs)`,
`to_dense_output(out)` → `[B, L_q, H_q, d]`, optionally `to_dense_inputs(inputs)` → dense
`(q, k, v)` for the correctness check, and set `kernel_regex` to match your kernel's name.
Find the name with

```
python -m kernelscope.run_kernel --plugin <name> --workload decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal --mode kernels
```

which lists every CUDA kernel one `run()` launches and which of them your regex
matched. Override `kv_heads_read(workload)` if your kernel expands K/V (no GQA), and
`supports(workload)` for anything else it cannot do (see `plugins/builtin/sdpa.py`).
Worked example of an external kernel taken verbatim: `plugins/builtin/triton_tutorial.py`
drives Triton's own `06-fused-attention.py` (v3.4.0, vendored under `samples/triton_tutorial/`).

**Standalone CUDA binary** — an `ExecutablePlugin`. Your binary takes the workload on its
command line, runs the kernel `iters` times timing each launch with CUDA events, prints one
line `KERNELSCOPE {"kernel_time_us": <median>, "launches_per_iter": N}`, and (when given an
output path) writes the result `[B, L_q, H_q, d]` as raw floats. The plugin supplies
`command(workload, iters, out_path)` and, for the correctness check, `reference_inputs(workload)`
rebuilding the binary's inputs in torch. Worked example: `samples/naive_attn/naive_attn.cu`
(`make` there) with `plugins/builtin/naive_exec.py`. In the simulator track the NVBit tracer
wraps the binary directly, so binaries get the full what-if treatment; on real hardware
timing is self-reported (no torch.profiler around a foreign process).

## What each track records (long format: workload_key, kernel, backend, metric, value)

| backend | metrics |
|---|---|
| `check` | max_abs_diff, ok |
| `latency` | median_s, min_s (CUDA events, launch overhead included) |
| `profile` | kernel_time_us, launches_per_iter; per launch: dur_us, grid_blocks, block_threads, regs, smem_bytes, sm_coverage, occupancy_device, ... |
| `analytic` | total_bytes, kv_bytes, flops, arithmetic_intensity, achieved_gbps, achieved_tflops, dram_util, tc_util |
| `trace` | warp_insts, est_sim_s, n_kernels (from the tracer's stats file) |
| `tracemix` | frac_global_mem / shared_mem / tensor / control / other, global_load_bytes, global_store_bytes |
| `sim:<variant>` | status, sim_wall_s, gpu_tot_sim_cycle, gpu_tot_ipc, gpu_tot_occupancy, L2_total_cache_miss_rate, ... |
| `ncu` (opt-in) | the fixed 10-metric set in `backends/realhw/ncu.py` |

## Running tests

```
/scratch/uceeeee/conda_envs/gradkernel/bin/python -m pytest -q            # everything (GPU tests included)
/scratch/uceeeee/conda_envs/aimers/bin/python -m pytest -q                # CPU-only subset
```

Always call interpreters by absolute path on this machine — `conda activate` does
not reliably put the env first on `PATH`.
