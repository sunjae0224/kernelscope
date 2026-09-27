# Follow-up for Codex: sim-track calibration after the 4090 gate (2026-09-19)

Your gate report (`docs/setup/accelsim_4090_gate_report.md`) was reviewed and reproduced:
69 accelsim/store tests and 190 CPU tests pass, `validation.csv` matches your table, and the
SM89 config's L2 (1024 sets x 128 B x 12 ways x 48 sub-partitions = 72 MiB) and DRAM
(24 x 16-bit at 21 GT/s = 1008 GB/s) derivations check out. Thank you for reporting the
out-of-target ratios plainly. Same ownership rules as the first handoff
(`docs/handoff/2026-09-18-codex-sim-track-4090.md`); keep working on `sim-track-4090`.

## What changed on the other side
The design for the rest of the project is `docs/plan/2026-09-19-design-surrogate-dispatcher.md`.
Two findings there bear on your validation:
1. **Your warm-cache diagnosis is confirmed from the real-HW side.** The real-HW track never
   flushed L2, so every hardware number you compared against was warm (the analytic track even
   reports 1762 GB/s for FD B16 L1K, above the 1008 GB/s DRAM peak). The other agent is adding a
   `--cache-state cold` mode to `run_kernel` (flushes 4 x L2 before each iteration with a kernel
   named `kernelscope_l2_flush`, excluded from kernel time) and will write cold and warm rows,
   distinguished by a `cache_state` column, to `results/hw_4090/`. **Do not build your own
   warm-up convention into the simulator; compare against the cold rows once they land.**
2. The FA2 split-KV heuristic and ragged batches matter for the project. You do not need to
   simulate ragged batches; uniform workload keys stay unchanged.

## Requests, in order
1. When `results/hw_4090/` has cold rows for your seven validation cells, recompute the
   sim/real table against **cold** `kernel_time_us`. Report both the old (warm) and new (cold) ratios.
2. Adopt `-gpgpu_kernel_launch_latency 0` as the documented convention **for comparisons with
   profiler kernel time** (profiler durations exclude host launch delay). Keep the 5000 value in
   a separately named config if you want an event-latency comparison.
3. Evaluate `-gpgpu_memory_partition_indexing 6` (IPOLY_MODULO) against mode 2 for the 48
   sub-partitions: slice-balance evidence (you already have the script), base-cycle change on the
   seven cells, and whether `sm_x2` stops *increasing* cycles. If mode 6 fixes the imbalance
   without breaking replay, make it the SM89 baseline in a new config revision and re-run the
   seven cells x seven variants. Never relabel old results; write to a fresh results directory.
4. Record the core clock used for cycle-to-time conversion in every sim row (for example an
   `arch` or `clock_mhz` extra column via `ResultStore.write(..., extra=...)`), so `report` can
   stop relying on the 1410 MHz default. Tell the other agent the column name in STATUS.
5. Optional, only if 1–4 are done: an `smem_x1.64` variant (100 KiB -> 164 KiB shared memory per
   SM, A100-like), because the design's key hardware argument is that the split-KV kernel's
   80 KiB CTAs fit once per SM on Ada but twice on A100.

Append results to `docs/STATUS.md` under your own dated heading and update the gate report's
validation section. Stop and report if a request would require editing files outside your area.
