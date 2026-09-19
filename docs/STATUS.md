# Status log

## 2026-09-01 (evening) — W1 D3–5 done: external kernels, w1 grid on both tracks, roofline

**Measurement hygiene incident.** GPUs 0 and 1 were 100 % busy with the user's own `eval_ruler.py`
jobs all day; every real-HW number taken on GPU 1 earlier today was under contention (large-KV cells
showed profiler kernel time 3× *above* CUDA-event latency — physically impossible). All real-HW
numbers below were re-measured on idle GPU 3. The sweep now records `gpu_util_at_start` /
`other_pids_at_start` on every row and prints a warning (`backends/realhw/hygiene.py`).

**Done (167 CPU + 25 GPU tests green):**
- **ExecutablePlugin path** — `samples/naive_attn/naive_attn.cu` (naive fp32 decode attention, one block per
  head, two passes) driven via the `KERNELSCOPE {json}` contract; correctness checked against the reference
  from a shared integer-hash init; traced by NVBit directly (no Python in the loop).
- **External Triton kernel** — Triton's own `06-fused-attention.py` (v3.4.0, vendored verbatim) registered as
  `triton_tutorial` (prefill, square, no GQA → KV expanded). flash_attn's bundled Triton kernels do not
  compile on triton 3.4.
- **Instrumentation-window tracing** — JIT/autotune warm-up is no longer recorded: the tracer starts with
  `NVBIT_INSTRUMENTATION_ENABLED=0` and `run_kernel --mode trace` toggles the injected tracer's
  `enable/disable_nvbit_instrumentation()` around one run (Accel-Sim's torch_hook mechanism; torch's
  `cudaProfilerStart` does *not* reach NVBit's ACTIVE_FROM_START=0 path). Verified: 3 warm-up launches
  recorded with 0 instructions, the windowed launch with 519,944. Stats and `kernelslist.g` are also
  filtered by the plugin regex post hoc.
- `kernelscope plot` — roofline from the analytic track (`results/roofline_w1.png`, 42 points).
- `report` verdicts now say "starved: grid covers 7 % of SMs, no resource helps" when nothing moves.

**Real-HW, idle GPU 3 (ceilings: HBM 1804 GB/s, fp16 GEMM 265 TFLOPS):**

| kernel | cell | kernel µs | CTAs | HBM util |
|---|---|---|---|---|
| fa2 | decode B1 L1K / L8K / L32K | 54 / 395 / 1557 | 8 | 4.3–4.8 % |
| fa2 | decode B16 L1K / L8K / L32K | 83 / 603 / 2383 | 128 | 45–50 % |
| flashdecoding | decode B1 L1K / L8K / L32K | 14.6 / 34 / 99 | 64 / 176 / 192 | 16 / 54 / 75 % |
| flashdecoding | decode B16 L1K / L8K / L32K | 57 / 323 / 1217 | 384 | 65 / 92 / **98 %** |
| naive_exec (fp32) | decode B1 L1K / B16 L8K | 904 / 13,900 | 32 / 512 | 0.5 / 4 % |
| triton_tutorial | prefill B1 L1K / L4K | 232 / 2452 | 512 / 2048 | 14 / 21 % of TC peak |
| fa2 / sdpa_cudnn | prefill B1 L4K | 851 / 849 | 1024 / 2048 | 61 % of TC peak |

**Simulation (Accel-Sim SM80_A100, 7 what-if variants, 6 cells finished so far):**

| kernel | cell | sim/real | bw_half | bw_x2 | sm_half | sm_x2 | verdict |
|---|---|---|---|---|---|---|---|
| fa2 | B1 L1K | 0.89 | +0.2 % | +0.1 % | −0.0 % | −0.0 % | starved (8 CTAs) |
| fa2 | B1 L8K | 0.87 | +0.5 % | +0.5 % | −0.0 % | −0.0 % | starved |
| fa2 | B16 L1K | 0.98 | +43 % | +5 % | **+79 %** | −18 % | parallelism-bound |
| flashdecoding | B1 L1K | 1.23 | +18 % | −5 % | +17 % | 0 % | bandwidth-bound (weak) |
| flashdecoding | B1 L8K | 1.37 | +50 % | −18 % | **+59 %** | +1 % | parallelism-bound |
| flashdecoding | B16 L1K | 1.25 | **+69 %** | −21 % | +18 % | +12 % | bandwidth-bound |

Reading: FA2 at B=1 is indifferent to *every* resource — it launches 8 CTAs and cannot use more
machine; no hardware change helps, only the kernel can. Once the grid fills the GPU (B=16) it becomes
parallelism-bound (halving SMs +79 %), and split-KV at B=16 is genuinely bandwidth-bound (halving HBM
+69 %), matching its measured 65 % HBM utilisation. L2 size never matters for decode attention (streaming).
Sim/real: 0.87–0.98 for FA2, 1.23–1.37 for split-KV (combine kernel over-estimated — W2 item).

**Pending / running:** B1 L32K and B16 L8K/L32K sim cells (budget-gated), naive_exec through the
simulator, Triton tutorial + fa2 prefill 1K through the simulator (base, sm_x2).

**Next (W2):** full decode grid on real HW; sim validation table (sim vs profiler kernel time, sim DRAM
bytes vs analytic bytes); investigate the split-KV combine-kernel over-estimate; flashinfer plugin;
heatmaps; write chapters 3–4.1.

## 2026-09-01 (later) — Nsight-free real-HW track is the default

ncu is blocked on this host by `dcgm-exporter` (root needed); the user chose Nsight-free as the default.
Real-HW track = CUDA-event latency + torch.profiler kernel time/geometry/occupancy + analytic bytes/FLOPs
vs measured ceilings + NVBit-trace instruction mix; `ncu` is opt-in. Ceilings gotchas: tensors > 2^31 bytes
hit torch's 64-bit-index path (halved bandwidth), single-launch L2 benches are launch-bound, the GPU
throttles at its 400 W cap on long GEMMs.

## 2026-09-01 — Day 1 (W1 gate + skeleton)

Repo skeleton (separate git, nothing committed); plugins sdpa_*/fa2/flashdecoding; **Accel-Sim W1 gate
PASSED** (NVBit 1.8 works under driver 595; tracer regex is `std::regex_match` on the mangled name →
wrapped as `.*(?:re).*`); ncu blocked by DCGM.


## 2026-09-18 — sim track (RTX 4090)

**D1 PASS, D2 functional PASS, D3 PARTIAL / BLOCKED; D4 defaults deferred.**
Work is on `sim-track-4090`. Full evidence, versions, commands, calibration
probes and remaining limits: [RTX 4090 gate report](setup/accelsim_4090_gate_report.md).

- Built pinned Accel-Sim v2.0.0, NVBit 1.8 and GCC 11 / CMake 3.31 / CUDA 12.9
  under `/home/skkai/accelsim/` and conda `accelsim-build`; system tools untouched.
  Driver 580.95.05 tracing passed. Unmodified SM86 gate: 524288 warp instructions,
  36589 cycles, clean exit. Native binary version 89 needed the documented Ampere
  opcode-map compatibility patch; this is not complete Ada ISA support.
- Installed SM89_RTX4090 resource config: 128 SMs, 72 MiB L2, 384-bit bus,
  2520 MHz core / 5250 MHz DRAM, 24 CTAs/SM. Native-sm89 vector-add and all seven
  variants passed. Parent HMMA/tensor-core settings were retained. The parent
  actually uses 16 x 16-bit controllers and CTA limit 32; derivation follows
  the pinned file rather than the anticipated parent values in the task.
- Created shared `gradkernel` with torch 2.8.0+cu128 / flash-attn 2.8.3.post1.
  The requested wheel required glibc 2.32; this host has 2.31, so the exact same
  version was rebuilt locally (1518.55 s, CUDA 12.9/GCC 11, sm80 SASS). Imports
  and four real-HW attention correctness checks passed.
- Completed smoke plus **28/28 attention variant replays** for FA2 and
  FlashDecoding B1 L1K/L8K. Real profiler / base simulator comparison:

| kernel | cell | real kernel us | sim cycles | sim/real at 2520 MHz |
|---|---|---:|---:|---:|
| fa2 | B1 L1K | 63.584 | 170012 | 1.061 |
| fa2 | B1 L8K | 494.336 | 1300168 | 1.044 |
| flashdecoding | B1 L1K | 10.016 | 39077 | **1.548, outside target** |
| flashdecoding | B1 L8K | 27.456 | 185450 | **2.680, outside target** |

FA2's 8-CTA B1 grid covers 6.25% of this GPU; both cells remain starved.
FlashDecoding is bandwidth-bound in the model (+41%/+89% for half bandwidth),
but its absolute timing is not calibrated. Isolated clock, DRAM latency and
zero-launch-delay probes are recorded in the report; no fit-only change was
promoted to the canonical config. Scoped warm-up experiments support cache-state mismatch: repeating just the
two captured FD launches with zero launch delay gives second-pair sim/real
ratios **0.831 (L1K), 1.054 (L8K)**. These exploratory results are separate from
the baseline parquet table. B16 and all variants still need validation under
an explicit warm-up accounting convention before adopting it.

B16 L1K and naive fp32 B1 L1K are **not yet measured/replayed**. After idle B1
measurements, foreign Python jobs resumed (PID 998528, then 1017901); the next
idle gate stopped at 71% utilization. The desktop rerun viewer remained present
during idle measurements and its warning is retained in the result records.
No foreign job was stopped. Naive's Makefile only gained the allowed sm89 gencode;
its binary built successfully. Per the requested deliverable ordering, old A100
backend/README defaults remain until D3 is completed. Validation scripts pass
this host's root, work directory, architecture and device explicitly.

Backend changes: preserve fractional DRAM clocks, detect unsupported binary 89
and subprocess errors/timeouts, select CUDA 12.9 cuobjdump instead of system 10.1,
isolate concurrent CPU variant artifacts (`--sim-jobs`, default 1), and expose
`--sim-rate`. Base FD L8K measured 11638 warp-inst/s; planning now uses 10000.
SMx2 measured 4493; use `--sim-rate 4000` for conservative seven-variant planning.
**69 simulation/store CPU tests pass.** Real SM89 stats/stdout fixtures are checked
in. The previously missing ResultStore was restored as explicitly authorized.

Result contract and `backend="sim:<variant>"` are unchanged. Analysis, real-HW,
plugins, run_kernel and pyproject were not edited. Always pass `--clock-mhz 2520`
to the shared report command; its default is still 1410 for A100.

Artifacts: `results/{hw_4090_simtrack,sim_4090}/20260918-validation2`; generated
table: `results/sim_4090/20260918-validation2/validation.csv`. The result directory
is ignored by git; trace/log provenance lives under `/home/skkai/accelsim/`.
Reproduce with `bash docs/setup/validate_4090.sh` once the GPU is idle; replay
saved scoped traces without the GPU using `python -m docs.setup.replay_4090`.


## 2026-09-18 — sim track (RTX 4090), resumed validation

**All requested executions completed; D4 host defaults enabled. Timing accuracy
remains outside target for FA2 B16 and FlashDecoding.** This supersedes the GPU
block and deferred-default status in the earlier entry without rewriting it.

The GPU was idle at 19:17 KST with only the desktop rerun viewer present.
B16 FA2/FlashDecoding and naive fp32 passed hardware correctness checks; their
scoped traces were saved before CPU replay. All **42 attention variants plus
one naive base replay** finished cleanly. An exact-key audit verified 43 unique
successful simulator results and seven hardware cells with no missing or
duplicate results. Final table: `results/sim_4090/20260918-complete/validation.csv`;
audit/provenance: `audit.json` and `audit.py` in the same directory.

| kernel | resumed cell | real kernel us | sim cycles | sim/real at 2520 MHz |
|---|---|---:|---:|---:|
| fa2 | B16 L1K | 64.303 | 276533 | **1.707, outside target** |
| flashdecoding | B16 L1K | 38.2235 | 294358 | **3.056, outside target** |
| naive_exec | B1 L1K, fp32 | 465.920 | 837308 | 0.713 |

Both B16 baseline models are bandwidth-bound (+97%/+102% for half BW).
FlashDecoding agrees qualitatively with the prior A100 verdict; FA2 differs
from A100's parallelism-bound result. Naive's reference time is self-reported
CUDA-event timing from the executable; attention uses torch.profiler.

Separate B16 zero-launch-delay plus scoped warm-up probes yielded ratios
1.470 (FA2) and 3.039 (FD), still outside target. Mode 2's 64-to-48 partition
hash reduction was found to bias 16 L2 slices: actual FD counters measured
2.003x more read events per slice there than in the other 32 slices. Thus a
nominal 72 MiB capacity does not establish correct cache residency. Mode 6
(IPoly-Modulo) is a candidate for a new controlled calibration revision, not
a validated replacement. The canonical config was kept fixed for this sweep;
no exploratory warm-probe numbers were substituted for baseline parquet rows.
See the [updated gate report](setup/accelsim_4090_gate_report.md) for all seven
validation rows, knob probes, raw evidence, remaining limitations and commands.

D4 now defaults to `/home/skkai/accelsim/accel-sim-framework`, work directory
`/home/skkai/accelsim/kernelscope_sim`, architecture `SM89_RTX4090`, device 0.
`ACCELSIM_ROOT` still overrides the root. README quick-start changes are limited
to host environment/device/path references. The shared report default remains
1410 MHz; **always pass `--clock-mhz 2520` for this model**. Analysis, real-HW,
plugins, run_kernel and the result contract remain untouched.

Resumed FD B16 base throughput was 7839 warp-inst/s and SMx2 was 3591;
planning now uses **5000**, with `--sim-rate 3000` for conservative planning
across the measured variants. CLI help reads the live planning constant.
The replay helper now supports `--plugins` to select saved traces and rejects
missing requested kernels. **69 simulation/store tests pass** after the D4
changes; CLI defaults, installed tool/config paths and the root override were
also checked successfully. The model is usable for explicit experimental
replay and is not yet a calibrated RTX 4090 performance predictor.

## 2026-09-19 — design-1-3: Phase 0 foundation

Phase 0 measurement campaign complete on the RTX 4090 (worktree `design-1-3`,
Task 12). GPU idle throughout (only the `rerun` viewer present); no hygiene
wait needed. `machines/rtx4090.json` measured: DRAM 952.6 GB/s, L2 plateau
4.85 TB/s, CTA DRAM/L2 26.0/46.4 GB/s, `block_placement.distinct_sms` 128 —
all matching the design spec's probe facts. New grids `grids/dispatch_s{1,2}.yaml`
and `grids/ragged_s{1,2}.yaml` (uniform and ragged-batch decode, S1/S2 head
geometries) drove 8 `bench` runs, 6801 cells total, **0 errors**, in ~10.6 min
of GPU time (well under the ~1 h estimate). All **five acceptance checks
pass**: uniform S1 dense cold regret 0.72 %/5.71 % (median/max), warm max
41.66 %; the ragged check cell's heuristic is 6.85× slower than the best
fixed split (required 5–10×); the worst ragged cell is 12.5× slower on the
dense path and 3.8× on the paged path; `iterations_dropped` ≤ 2 for 100 % of
ok cells everywhere, `check_ok` false nowhere. Full numbers, per-B fa2
crossover, and the ragged worst-case dense-vs-paged comparison (a fact beyond
spec F14, which was dense-only) are
in [docs/plan/2026-09-19-p0-campaign.md](plan/2026-09-19-p0-campaign.md).
