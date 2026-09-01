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
