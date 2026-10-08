# Status log

## 2026-10-08 — FlashInfer-in-engine campaign re-measured, kernel probes (64K and FlashInfer), library-agnostic table policy (branch `design-1-3`)

- **FlashInfer inside the engine, re-measured** (TODO §1 item 1; the 2026-10-07 queue had deleted its result folders): `serve_4090/flashinfer_20261006/{ragged,uniform,arrivals}`, Qwen3-4B, `--policy-cache keep`, 3 repeats, KV 10 GiB, with the GPU test of the adapter (fixed 10-08) passing 10/10 before the runs and only the resident rerun viewer on the GPU. Ragged: heuristic 60.99 ms → table 33.70 (1.810×), `flashinfer_cudacore` 33.28 (**1.833×**; attention 8.69 vs 9.23 ms/step, plan() 370.8 µs/step), `flashinfer` tensor-core 48.51 (**1.257×**); uniform all four within ±1 % (plan() 291–338 µs/step outweighs the slightly faster kernels); arrivals cudacore 1.181× ≈ table 1.173×, tensor-core 1.092×. The 10-07 log summary matched. Outputs: FlashInfer policies are not token-identical to the FA2 heuristic (39–223 differing tokens); teacher-forced diagnostics for uniform/arrivals were added (16:28–16:33) so every event is classified — mostly tie_1ulp, 2 tie_2ulp (uniform cudacore) and a single clear position (rid 8, position 41, 70 ulp) in ragged and uniform for both backends, the same event as the 10-02 control (same prompt hash). Note [§4](experiments/2026-10-06-problem-scope.md), verify `fiengine.*` (148 checks).
- **Kernel probes** — `scripts/probe_split_kernel.py` generalized (argparse; GQA float32 reference without expanding K/V to the query heads; the target row by batch index; `--candidate-backend {fa2,flashinfer,flashinfer_cudacore}`; 17 CPU tests). (a) The 64K clear event (rid 0, position 27, split 1 vs 8, KV 13 GiB): every layer's split-1/split-8 outputs within one bf16 spacing of the layer's largest output (max 0.125 at |out| 31.8; vs float32 0.123) → not a kernel error; but unlike 10-02 the single step on the identical KV state flips the argmax (220 → 15, |Δlogit| 8.99, cosine 0.10; the 31 short requests identical) — amplification within one step through 36 layers in a 65K random-token context. Note §3c, verify `serve64k.probe_*` (7). (b) The FlashInfer clear event (ragged, rid 8, step 40) for both backends: max |backend − split1| 0.25 = one spacing at |out| 33.13, 0/36 layers over, argmax kept on the identical KV state (|Δlogit| 0.31 cudacore / 1.97 tensor-core), 0 argmax changes over 32 requests → not a kernel error; the 22.x teacher-forced gap is accumulated KV rounding as on 10-02. Note §4b.
- **Library-agnostic table policy** (TODO §1 item 3; design [docs/plan/2026-10-08-design-anytable-policy.md](plan/2026-10-08-design-anytable-policy.md), implemented test-first by a sonnet subagent and reviewed): `demo_data/dispatch_paged_cold_any.csv` (`static_default_losses` over the six 32K grids: 211 complete cells; overall best CUDA-core 148 / tensor-core 45 / FA2 18), `TablePolicy(csv, backends, plan_us)` picks (split count, backend) per step with cost = n_layers × kernel_us + plan_us for a FlashInfer backend, `--policy table_any:<csv>[:<plan_us>]`, the engine records the backend per step (`steps.parquet` column `attention`, right after `num_splits`; existing columns unchanged), `package_demo.py --tables-only`. Campaign `serve_4090/anytable_20261008` (heuristic / table / table_any / table_any_p371, 371 µs = the measured cudacore plan cost): ragged `table_any` chooses CUDA-core on every step and lands where the fixed policy did (33.26 ms, **1.831×** vs table 1.810×, +1.15 %, faster in all 3 repeats), `table_any_p371` returns to FA2 split 16 (1.806× = table); uniform all three tables choose FA2 split 0 (1.00×); arrivals table_any 1.178× ≈ table 1.174× (+0.36 %, about the noise floor; p371 1.172×). `table_any`'s token histories are identical to §4's cudacore run in ragged and arrivals (0 differing tokens, every repeat), so its events are the same ones (ragged 3 tie_1ulp + the probed clear; arrivals 6 tie_1ulp). Conclusion (PLAN §2 8): letting the policy also switch libraries adds ≈1 % on ragged batches at the cost of output identity with FA2; the per-step plan() cost, the same size as the kernel gain, is the limit. Note §5, verify `anytable.*`.
- Docs: PLAN §1 ⑤/⑦ and §2 2/4/8, note §3c/§4/§5, README (`table_any`, verify count), TODO rewritten. Tests: `CUDA_VISIBLE_DEVICES= make test` 667 passed (5 skipped); `make verify` 531/531 PASS against the bundle.
- Host notes: all GPU work ran 16:01–17:04 through idle-gated queue scripts (8 × 30 s polls); a `serve run` exit code 1 means strict token equivalence failed — expected for FlashInfer policies — and queue scripts must never treat it as a failure or delete the folder (the 10-07 mistake).

## 2026-10-06 (afternoon) — Problem scope: trace replay, two libraries' static defaults, 64K, FlashInfer inside the engine (branch `design-1-3`)

Answer to "is the problem too narrow and the gain too small?" — note [docs/experiments/2026-10-06-problem-scope.md](experiments/2026-10-06-problem-scope.md), review addendum [§5](plan/2026-10-06-review-problem-definition.md), PLAN §0 rewritten.

- **Trace replay (GPU-free)** `kernelscope/analysis/traffic.py` + `scripts/traffic_replay.py`: public request traces (Azure LLM Inference 2023 conv/code, BurstGPT v2) replayed through the engine's admission rule at decode-step granularity and every step scored with the fitted surrogate (heuristic vs best split). Azure conv at one-4090 load (batch mean 34.8, 89.5 % of steps past the no-split threshold B ≥ 26): **13.5 %** of steps lose ≥ 1.25×, attention-time ratio **1.119**; Azure code ×16: 14.1 % / 1.136; BurstGPT (prompt median 502): **0.3 %** / 1.018. What-ifs: prompts ×2 at 10 GiB KV 25.7 % / 1.156; prompts ×4 at 10 GiB shrinks the batch to 14.6 (< 26) and the loss vanishes (2.9 %), while prompts ×4 with 436K-token KV keeps batch 34.8 and reaches **39.5 % / 1.222**. Conclusion: a conditional problem (long-prompt share × batch ≥ 26), not a rare one. 20 replays bundled as `demo_data/traffic/<run>/summary.json`.
- **Two libraries' static defaults** `analysis.dispatch.static_default_losses/_summary`: on the same paged grids, loss of each default against the best of all variants (FA2 splits + both FlashInfer variants). 32K ragged (162 cells): FA2 heuristic median 1.457×, FlashInfer tensor-core (GQA default) median 1.017×, p90 1.489×, max **2.485×**, 36 cells > 1.25×; FlashInfer CUDA-core max 1.014×; the project's FA2-only dispatcher max **1.063×**. Uniform: every default ≤ 1.171×. 64K ragged (18 cells): heuristic median 2.861×, max 5.265×; tensor-core median 1.634×, max 2.989×.
- **64K long side**: grids `ragged_s1_64k` / `uniform_s1_64k` (20 cells × 11 variants, 0 errors). Matched cells (18 per length): heuristic / best FA2 split median 8K 1.545 → 16K 1.933 → 32K 2.558 → 64K 2.839, max 5.183; FlashInfer tensor-core 1.126 → 1.634; CUDA-core ≤ 1.014 at every length. Full model `scenarios/graduation_ragged_64k.yaml` (65536×1 + 512×31, KV 13 GiB, `--policy-cache keep`, 3 repeats): heuristic TPOT **96.93 ms** (attention 82.8 % of the step) → table **40.06 ms (2.419×)**, model 2.463×, hybrid **2.463×**; all three choose split 8 at every step. One divergence event, identical across policies and repeats (64K request, position 27, 4 of 2,048 tokens, re-joins); teacher-forced classification queued (§3c).
- **FlashInfer inside the engine** `kernelscope/serve/attention.py` (`FlashInferDecode`: drop-in for `flash_attn_with_kvcache` on decode steps — appends the step's K/V into the paged pool, `plan()` once per step before the layer loop, `run()` per layer; prefill stays flash-attn). Policies `flashinfer` / `flashinfer_cudacore` (`FixedPolicy(..., attention=...)`), the engine books `plan()` time in `policy_us`. CPU tests with a fake wrapper; GPU test `tests/test_serve_attention_gpu.py` and the three-scenario campaign `serve_4090/flashinfer_20261006` are queued behind the shared GPU (results → note §4).
- Tooling: `package_demo.py --exclude '*_failed_*'` (default) and `traffic/*/summary.json` in the bundle; `./run.sh traffic` (downloads the Azure trace, replays at `RATE_SCALE`) and `./run.sh defaults`. Verify: 80 more checks (`traffic.*`, `defaults.*`, `longctx.*`, `serve64k.*`), `make verify` 191/191; `make test` 630 passed.
- **2026-10-07 night, detached GPU queue** (ran while no session was open): the 64K teacher-forced diagnostic succeeded on the third attempt (an ollama server kept taking the GPU) and classifies the single 64K event as **`clear`, 11 ulp, reproduced** — kernel probe pending (note §3c). The FlashInfer engine campaign ran for all three scenarios, but the queue's retry logic treated `serve run`'s exit code 1 (strict token equivalence fails for the FlashInfer policies, an expected outcome) as a failure and deleted the result folders; only the printed summaries survive in the log (note §4, marked preliminary: ragged flashinfer_cudacore 1.83×, tensor-core 1.26×, plan 370–430 µs/step; uniform all 1.00×; arrivals 1.18×/1.10×). Re-measurement is TODO §1 item 1. A sonnet review of the new modules found a spurious decode step for single-token requests, an infinite loop on NaN arrivals and a page-reservation formula off by one; all fixed with tests and the 20 replays re-run (headline numbers unchanged to three decimals).
- Host notes: the lab HDD was saturated by another user's 17 GB download, so the 8 GB checkpoint loaded at ~4 MB/s (10 min); an ollama server came and went on the GPU twice — one serving run was stopped before it wrote results and restarted after 8 idle polls.

## 2026-10-06 — Demo page and token race (branch `design-1-3`)

- **Demo page** (`dashboard/demo_page.py`, `kernelscope/dashboard/{demo.py,race.html}`, `scripts/export_race.py`, `run.sh demo-record`): the dashboard's first screen is a fixed five-scene flow (conclusion, race, why, how, verification); the previous five tabs moved to the Lab page. Live-measure and plugin-bench panels run on the GPU host; `export_race.py` writes `race.json` next to a recording so the race replays without the tokenizer. Design `docs/plan/2026-10-06-design-demo-page.md`; walkthrough `docs/demo.md`.
- **Recording `demo_text_20261006/ragged`** (natural text, 32K x 1 + 512 x 31, 64 tokens; heuristic / table / hybrid, 3 repeats, `--policy-cache keep`): TPOT mean heuristic **61.18 ms** (1.00x), hybrid **34.07 ms** (**1.80x**), table **34.08 ms** (**1.79x**); `tokens_equivalent` True for all three policies, output validation passed.
- **Race rule**: the two panels replay sequential measurements on one clock (zero = end of all prefills); concurrent execution would contaminate the measurement, so the page labels it "replay", not live.

## 2026-10-06 — Policy-cache protocol, engine guard, FlashInfer kernel grids, Policy lab tab (branch `design-1-3`)

- **`serve run --policy-cache {fresh,keep}`** (default `fresh` = the previous protocol). `keep` builds the policies once before warm-up and reuses them across repeats, so every page configuration the warm-up saw is a cache hit (steady-state optimistic bound; `fresh` is the pessimistic bound). Re-measuring `heldout_text_arrivals` with `keep` (`demo_data/serve_4090/hybrid_keepcache_20261006/`): hybrid **31.62 ms, 1.090×** (table 1.086×, model 1.091×, heuristic 34.46 ms), selection cost 14.95 µs/step, 0 cache misses, divergence events unchanged (2 per policy, all `tie_1ulp`). The 2026-10-02 ranking (hybrid 1.020× < table 1.085×) was the first-decision cost of the fresh protocol; with the cache kept the better kernel choices show in TPOT. Note [§3b](experiments/2026-10-02-hybrid-validation.md).
- **Engine guard**: `Engine.run` now refuses the `hybrid` policy outside d=128 fp16/bf16 like `model`/`table` (test added).
- **FlashInfer kernel grids** (plan 2026-09-26, kernel part): plugins `flashinfer_paged` (tensor-core, recommended for GQA) and `flashinfer_paged_cudacore` in `kernelscope/plugins/builtin/flashinfer.py`, measured cold on `ragged_s1` (162 cells) and `dispatch_s1` (49 cells); flashinfer-python 0.6.13 with the cu128 prebuilt JIT cache (the host's system nvcc is CUDA 10.1, so JIT compilation failed; that attempt is kept as `hw_4090/*_failed_jit_20261006`). Ragged: CUDA-core variant / best FA2 split median **0.977×**, max **1.003×**; tensor-core variant median 0.998× but up to **2.401×** slower (13 cells > 1.5×, e.g. the 32K+512×31 cell: 670.9 µs vs best FA2 `fd_s16_paged` 279.4 µs vs CUDA-core 269.9 µs). Uniform: faster variant median 0.998×, CUDA-core up to 1.181×. Conclusion recorded in PLAN §2 item 4: choosing the split externally recovers essentially all of the kernel-swap gain; FlashInfer itself needs a variant choice. Remaining: `serve run --attention flashinfer`, plan() cost. Note [§8](experiments/2026-10-02-hybrid-validation.md).
- **Dashboard 05 Policy lab** (`kernelscope/dashboard/lab.py`, tab in `dashboard/app.py`): side-by-side protocol comparison (fresh/keep), FlashInfer vs best-FA2 per cell with both variants, divergence-event overview, and a command composer for `serve run` / `bench` that can launch the run on the GPU host after `serve doctor` and tail its log. Data functions tested without Streamlit; app smoke tests updated to five tabs.
- **vLLM reproduction** (PLAN §2 item 7): separate environment `~/.venvs/kernelscope-vllm` (vllm 0.31.0, torch 2.13 cu130; the project env stays on torch 2.8) and `scripts/vllm_reproduce.py` (decode time per step from two generation lengths, FLASH_ATTN vs FLASHINFER, eager mode). Result (note §9): the ragged batch is 5.46× slower per step than the uniform batch in vLLM 0.31 too, but identically for FLASH_ATTN and FLASHINFER (0.997×), with CUDA graphs and with KV block 256 — so the slowdown there is not the split heuristic this project explains; profiling a vLLM step is the next item. First attempt (no warm-up, prefix caching on) kept as an invalid record.
- Verify: 25 more checks (`hybrid.keep_*`, `divergence.keep_*`, `flashinfer.*`, `vllm.*`), `make verify` 111/111. Problem-definition / related-work review in `docs/plan/2026-10-06-review-problem-definition.md`.

## 2026-10-02 — Hybrid policy in real generation, divergence metric replaced (branch `design-1-3`)

- **Hybrid policy validated in generation** (PLAN §2 item 1; `serve run --policy hybrid:demo_data/dispatch_paged_cold.csv:0.2`, Qwen3-4B, protocol of `graduation_20260922`, 3 repeats, commit 195e9e1 with a clean tree): on the three requested scenarios the hybrid policy picks exactly what the table policy picks on every step (0/567 steps differ — the scenarios sit on measured table cells), so TPOT differs only by the selection cost: ragged **61.11 → 34.09 ms (1.793×**, table 1.807×), arrivals 1.146× (table 1.164×), uniform 0.993×; tokens identical in all nine runs; heuristic/table TPOT within 1.25 % of the September campaign. Supplementary run on `heldout_text_arrivals` (the one recorded condition where a GPU-free replay shows hybrid ≠ table): hybrid follows the model ranking on 12/55 steps, attention 3.31 → 3.00 ms/step (model 2.93), but 13 cold decisions per run (1.56 ms/step under the "fresh policy cache per run" protocol) leave TPOT at 1.020× vs table 1.085× and model 1.028×. Bundle `demo_data/serve_4090/hybrid_20261002/`, note [2026-10-02 hybrid validation](experiments/2026-10-02-hybrid-validation.md).
- **Divergence metric replaced** (PLAN §2 item 2): `scripts/check_policy_numerics.py` now records the top-1 logit values, `kernelscope.serve.divergence.classify_events/campaign_events` join free-generation divergence events with the teacher-forced rows (margin = reference gap to the token the candidate actually chose, in bf16 spacings of the reference top logit), and `scripts/classify_divergence.py --campaign <dir>` writes `divergence.csv` and exits non-zero on `clear`, unclassified, unreproduced, missing-token or outside-event cases. Results: requested scenarios 0 events / 0 teacher-forced flips; held-out arrivals 2 events per policy, all 6 `tie_1ulp` with measured top logits 31.25–38.75 (the September "tie_2ulp" labels came from the assumed logit range); control `graduation_uniform × fixed:8` 6 events (same positions as September): 5 `tie_1ulp`, **1 `clear`** (request 8 position 41, 70 spacings, max |Δlogit| 22.625).
- **The clear event is not a kernel error**: `scripts/probe_split_kernel.py` replays the reference history on the identical KV state and compares every layer's attention output for num_splits 1 and 8 against a float32 reference — both within 0.134 of the reference (bf16 output rounding, identical for the two splits), split-1 vs split-8 at most 0.0625 (one spacing), and a single step computed with split 8 on the identical state keeps every argmax (request 8: max |Δlogit| 1.08). The 22.625 gap is the amplification of rounding differences the fixed:8 run accumulated in its own KV cache over 40 steps, surfacing where an attention-level copy choice is tied. Consequence recorded in PLAN §1 ⑤: divergences are not only logit ties; `clear` stays a "investigate" flag, decided by the kernel probe.
- Also in the ragged teacher-forced diagnostic: the 32K-token request's logits drift up to 5.41 at late steps under independent KV evolution while 96.9 % of positions are bit-identical and no argmax changes; same mechanism, out-of-distribution synthetic prompt.
- Verify: 39 new `hybrid.*` / `divergence.*` checks recompute every number above from the bundle (summaries from parquet, events from token histories + teacher CSVs, probe CSVs); `make verify` **86/86**. CPU suite 550 (527 + 23 new tests).

## 2026-09-27 — Op-class breakdown of the decode step (branch design-1-3)

- **`serve diagnose` complete** (spec `docs/plan/2026-09-27-design-op-breakdown-diagnose.md`, plan `docs/plan/2026-09-27-plan-op-breakdown-diagnose.md`; identity unchanged — cause-explanation stage, not a profiler): `OpTimer` (8 op classes: embed, norm, qkv_proj, rope, attention, o_proj, mlp, lm_head) on top of `Engine.run(ops_mode="event")` alongside an untimed control run (`ops_mode=None`); `kernelscope/diagnose/opmodel.py` (torch-free compulsory bytes/FLOPs per class), `kernelscope/diagnose/report.py` (`diagnose()` recomputes verdicts memory_bound/compute_bound/launch_bound/parallelism_candidate/below_ceiling_unknown against DRAM/tensor-core ceilings at threshold θ=0.7, plus attention's chosen-vs-best variant, regret and Amdahl step bound against the nearest measured dispatch-table cell), `kernelscope/diagnose/figures.py` (`docs/img/op_breakdown_ragged.png`), `kernelscope/diagnose/run.py` + `serve diagnose` / `serve diagnose-report` CLI subcommands. `MachineSpec.tc_tflops` + `ridge_flop_per_byte()` added; `machines/rtx4090.json` recorded with it.
- **GPU runs D1–D3** (Qwen/Qwen3-4B-Instruct-2507, bf16, RTX 4090, `--kv-gib 10 --warmup-runs 1 --warmup-steps 2`; bundle `demo_data/serve_4090/diagnose_20260927/{ragged,uniform,heldout_ragged}/`): D1 `graduation_ragged` (32768×1+512×31, 63 steps) — attention share **70.9% (heuristic) → 37.5% (table)**, remaining GEMM classes already at 63–90% of the (lower-bound) DRAM ceiling (o_proj/mlp/lm_head `memory_bound`, qkv_proj ~63% below θ=0.7). D2 `graduation_uniform` (512×32, 63 steps) — attention share only **19.2%**, both policies pick the same variant, `memory_bound` under both (no policy difference, matching the earlier "no gain on uniform batches" result). D3 `heldout_text_ragged` (natural text, 12288×1+384×27, 31 steps, heuristic vs `model` policy) — attention share 49.8% → 23.7%.
- **A1–A8 acceptance**: A1 unattributed ≤10% PASS (max 2.82%). A2 timer overhead ≤5% PASS (max 4.19%). A3 D1 heuristic attention row matches the dispatch table cell exactly (regret 2.8363 == table's heuristic_regret). A4 only attention gets `parallelism_candidate` under the heuristic policy (D1, D3); it becomes `memory_bound` under table/model (D1 83.3%, D3 76.1% DRAM) and under both policies in D2 (75.7%). A5 measured control-step ratio slightly **exceeds** the Amdahl estimate (D1 2.139 vs bound 2.102; D3 1.523 vs bound 1.474) — D1 because in-situ attention speedup (3.98×) beats the table's cold-regret estimate (3.836×), D3 because the nearest measured cell is a distance-0.54 neighbour; recorded as-is, the bound is not claimed to be a strict upper bound. A6 `tokens_consistent` True on all three runs. A8 no performance-path regression: `serve run` step_us median unaffected by the diagnose changes, 18021.28 → 18028.05 (**+0.04%**, ≤2% budget). Full tables and reproduction commands: [2026-09-27 op-class breakdown](experiments/2026-09-27-op-breakdown.md).
- **Verify**: 7 new `diagnose.*` checks recompute the headline shares/ceilings/Amdahl-bound/overhead straight from the recorded parquet files via `kernelscope.diagnose.report.diagnose()` (never `diagnosis.json`); `make verify` now **47/47**. CPU suite green.
- **Limitations** (documented, not fixed here): the byte/FLOP model counts compulsory traffic only, so `pct_dram`/`pct_tc` are lower-bound estimates; kernel-internal behaviour (stalls, bank conflicts) is Nsight Compute's domain and outside what CUDA-event op timers can see; op timers themselves add host overhead, so the event run's step time is diagnostic only — the control run is the performance evidence; qkv_proj sits just under θ=0.7 in every run and o_proj straddles it (66–71%).

## 2026-09-27 — Op-class step breakdown (Tasks 1–5), midterm report v4 (branch `design-1-3`)

- **`serve diagnose` groundwork** (spec `docs/plan/2026-09-27-design-op-breakdown-diagnose.md`, plan `docs/plan/2026-09-27-plan-op-breakdown-diagnose.md`; identity unchanged — this is the cause-explanation stage, not a profiler): `OpTimer` generalises `AttentionTimer` to 8 op classes (embed, norm, qkv_proj, rope, attention, o_proj, mlp, lm_head) with the attention-only subclass keeping the frozen `start(layer)/stop(layer)/total_us()` API; `Engine.run(ops_mode="event")` returns per-(phase, step, rid, layer, op_class) rows in `RunResult.ops` while `ops_mode=None` is byte-for-byte the old path; `MachineSpec.tc_tflops` + `ridge_flop_per_byte()`; `kernelscope/diagnose/opmodel.py` (compulsory bytes/FLOPs per class, torch-free) and `kernelscope/diagnose/report.py` (verdicts memory_bound / compute_bound / launch_bound / parallelism_candidate / below_ceiling_unknown, unattributed and timer-overhead shares, attention row against the nearest measured cell, `diagnosis.json`/`ops.csv`). CPU suite 518 passed; `verify` still 40/40. Remaining Tasks 6–9 (figure, CLI, GPU runs D1–D3, verify items + docs) and three open review findings are listed in `TODO.md`.
- **Midterm report v4** (`docs/report/midterm/중간보고서_이선재_v4.pdf`, sources in `docs/report/midterm/src/`): table 7 V1·warm hybrid cell corrected from "미평가" to the recorded 41.7% with a sentence on the warm failure; chapter numbers restored so the "2장~5장" and "4장 라)" references resolve; abstract qualifies 12.53× as the contiguous-KV kernel figure; 6,801 defined as workload × cache state × candidate cells; limitations and table 8 gain the vLLM reproduction item; formal wording pass. All numbers re-derived from `demo_data/` (verify 40/40) before the rebuild.
- GPU note: the 4090 was occupied by another project's vLLM processes during this session, so the A8 performance-path baseline (pre-change `serve run` step time, commit 6ce775f) is still to be measured back-to-back with the post-change run.

## 2026-09-26 — Hybrid selection, mismatch root cause, GPU-free verification (branch `design-1-3`)

- Identity fixed for the midterm report: *measurement- and simulation-driven dynamic selection of the attention kernel split*. Profiler and simulator are pipeline stages, not products. Root `PLAN.md` v3 and `CLAUDE.md` rewritten; the July GPGPU-Sim plan is archived.
- `kernelscope verify` (new): recomputes **40 documented numbers** from `demo_data/` and `docs/experiments/` without a GPU and checks the documents still state them; `make figures` renders the report figures from the same data. Fixed CPU-only collection (`bench/stream.py` imports triton lazily) and made `dispatch-table` / `model validate` read the portable `summaries.jsonl` bundle.
- **Hybrid policy** (`HybridPolicy`, `--policy hybrid:<table>:<delta>`): model ranking, but the nearest measured cell decides when the model predicts it within δ. Leave-one-out replay on the recorded cells (`scripts/evaluate_hybrid.py`): δ = 0.2 passes the 5 % / 15 % selection criterion on all three cold sets (max regret **1.97 % / 7.54 % / 10.90 %** for V1 / V5 / V6) where the model alone (24.3 / 66.3 / 24.5 %) and the table alone (50.4 / 66.1 / 100.7 %) fail. Warm V1 still fails for every policy. Not yet run on the GPU. [Details](experiments/2026-09-26-hybrid-policy.md).
- **Mismatch root cause** (`scripts/analyze_mismatches.py`): every token disagreement in the 108-run follow-up starts at the same position in all repeats; the 10 failing runs contain 25 divergence events (8 at the last token). In the teacher-forced diagnostic all 4 argmax flips have a reference top-1 margin of 0.125–0.25 = 1–2 bf16 spacings, and the free-generation divergence positions coincide with those flips 4/4. Output validation will report divergence events classified `tie_1ulp` / `tie_2ulp` / `clear` instead of token agreement; the "restrict splits" idea from the midterm plan is dropped as it does not match the cause. [Details](experiments/2026-09-26-mismatch-analysis.md).
- FlashInfer comparison designed, not run: [plan](plan/2026-09-26-plan-flashinfer-comparison.md). CUDA 11.8 toolkit installed and GPGPU-Sim 4.2 built and running on the GPU-less laptop (`experiments/gpgpusim/`, recipe: uncompressed fatbin, `CUOBJDUMP_SIM_FILE=1`, no `-lineinfo`, no warp shuffles); a reduced fp32 split-KV decode kernel (CPU reference match 2e-8) swept over the split count reproduces the mechanism qualitatively: ragged 2048+128×15 is 8.5× slower than a uniform batch of similar key count at S=1 (IPC 105 vs 921) and speeds up **7.23×** by S=16, while the uniform batch saturates at 2.25× (S=8). Numbers in `docs/experiments/gpgpusim-splitkv-sweep.csv`, checked by `verify` (`sim.*`, now 40 items).
- CPU suite on the laptop: 470 passed before this entry's additions; new tests for verify, divergence, hybrid, evaluation scripts and report figures.

## 2026-09-22 — Follow-up: lower selection cost and two-model natural-text validation

- Optional native C event simulator retains the NumPy oracle and explicit compiler fallback. Seven old configurations have identical predictions/rankings; ragged cold selection **842.600 → 14.293ms**, with one-time build **55.683ms** disclosed separately.
- Added original natural-text scenarios, fixed resolved tokens, prompt/tokenizer hashes, BF16/backend metadata and setup timing. Existing table/fit parameters remain fixed.
- Two models × three conditions × two seeds × three policies × three repeats: **108 final GPU runs** after full-scenario warmup. Separate **108-run partial-warmup pilot** is preserved. All paired pilot/final generated-token arrays match.
- Model policy ragged TPOT improved **1.282–1.290× Qwen4B**, **1.184–1.188× Llama8B**, with exact generated tokens. Uniform has no practical gain. All arrivals comparisons and two ragged table comparisons fail strict equivalence: **14/24** non-reference policy/condition comparisons pass output validation.
- Teacher-forced natural arrivals diagnostic: 55 decode calls, 868 comparisons per policy, 866 argmax matches per policy, all finite. Does not override failed free-generation validation.
- Dashboard separates campaign/model/seed/source/warmup identities; shows first decision, later misses and hits, and masks invalid speedups. CSV/PNG/SVG reports and portable raw evidence included.
- **462 CPU tests passed, 4 skipped, 12 GPU tests deselected**. Offline wheel contains matching C source and no compiled `.so`. Independent Llama HF oracle passed. [Results and scope](experiments/2026-09-22-followup.md).

## 2026-09-22 — Graduation demo and whole-model serving evidence

- Added local Llama/Qwen3 paged decoder, deterministic continuous batching, heuristic/fixed/table/model policies, strict token checks, numeric diagnostics and local text generation.
- Four-view Korean demo: Diagnose, Kernel map, What-if and Serving, with explicit measured/predicted evidence, repeated results and portable recorded artifacts.
- Qwen3-4B / RTX 4090: 3 scenarios × 4 policies × 5 repeats. Table TPOT improved **1.806× ragged**, **1.174× arrivals**, with identical greedy tokens in all repeats; uniform table had no material improvement.
- Preserve negative results: uniform fixed8 token agreement 94.43%; model first-choice cost hurt uniform/arrivals latency. The campaign completes all independent scenarios and returns nonzero for the recorded equivalence failure.
- CPU surrogate validation passed timing-error thresholds but failed worst-case selection-regret thresholds; what-if remains experimental.
- Final CPU suite: **392 passed, 4 skipped, 12 GPU tests deselected**. New decoder/KV GPU tests: **7 passed**; independent full-Qwen numeric oracle and same-history policy diagnosis saved under `docs/experiments/`. Portable bundle: 272 copied files, all source hashes verified.
- Reproduction and limits: [serving results](experiments/2026-09-22-serving-results.md), [project scope](graduation.md), [demo script](demo.md). New code lives in worktree `kernelscope-design`.

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
