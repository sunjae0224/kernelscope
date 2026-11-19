# kernelscope

**GPU 프로파일링·시뮬레이션 기반 LLM 추론 attention 커널 동적 선택 시스템.**

KernelScope는 길이가 다른 요청이 섞인 LLM 디코드 배치에서 FlashAttention의 KV 분할 수(num_splits) 휴리스틱이 GPU를 채우지 못하는 문제를 RTX 4090 실측으로 진단하고, 성능 모델·Accel-Sim으로 설명·예측하며, 매 단계 분할 수를 동적으로 고르는 정책을 실제 Qwen3/Llama decoder에서 검증하는 졸업 프로젝트입니다. 프로파일러와 시뮬레이터는 이 파이프라인의 단계이며 제품이 아닙니다. **[프로젝트 설명과 연구 질문](docs/graduation.md)** · **[5분 데모 안내](docs/demo.md)**

## 졸업 프로젝트 데모

초기 합성 토큰 실험(RTX 4090 + Qwen3-4B)에서 실측표 기반 선택은 **혼합 길이 배치의 TPOT 61.06 → 33.80ms(1.81배)**, **요청 합류 시나리오 59.09 → 50.34ms(1.17배)**를 기록했습니다. 각 5회 반복에서 해당 정책의 생성 토큰은 기존 휴리스틱과 모두 같았습니다. 균일 배치에서는 개선이 거의 없었으며, 모델 기반 선택의 초기 계산 비용과 고정 split의 출력 불일치 사례도 보고합니다. [전체 결과](docs/experiments/2026-09-22-serving-results.md) · [발표용 비교 그림](docs/img/serving_comparison.png)

**후속 검증도 완료했습니다.** 모델 선택의 첫 계산은 기존 혼합 배치에서 **843 → 14ms**로 줄었습니다. 두 모델·두 시드의 새 자연어 조건에서 최종 108회 측정한 결과, 모델 정책은 혼합 길이에서 **Qwen 4B 1.28–1.29배, Llama 8B 1.18–1.19배** 빨랐고 출력도 같았습니다. 균일 배치의 개선 부재와 요청 합류 조건의 출력 불일치도 보존했습니다. [후속 결과와 해석](docs/experiments/2026-09-22-followup.md) · [36개 정책별 결과와 그림](docs/experiments/followup/README.md)

```bash
cd /home/skkai/AI_Accelerator/kernelscope-design
bash scripts/demo.sh
# http://localhost:8501
```

첫 화면 **Demo**는 결론 → 경주(두 정책의 기록을 타임스탬프대로 동시 재생) → 왜 → 어떻게 → 검증의 다섯 장면이고, 사이드바 **Lab**이 병목 진단, 조건별 최적 커널, 가상 하드웨어 예측, 실제 생성 실험, Policy lab의 다섯 탭입니다. 레이스용 자연어 기록은 `./run.sh demo-record`(GPU, 약 2분)로 만듭니다. 발표 순서는 [docs/demo.md](docs/demo.md). 저장된 결과를 탐색할 때 GPU는 필요하지 않습니다. 원본 결과 폴더가 없는 환경에서는 `demo_data/`의 실측 요약을 사용합니다. `KERNELSCOPE_RESULTS=/path/to/results`로 결과 폴더를 지정할 수 있습니다.

### 실행 환경

현재 장비에서는 기존 CUDA 환경을 유지하는 프로젝트 전용 `.venv`를 사용합니다.

```bash
/home/skkai/miniforge3/envs/gradkernel/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r env/requirements-demo.txt
.venv/bin/python -m kernelscope.cli serve doctor
```

다른 장비에서는 해당 GPU에 맞는 PyTorch·flash-attn 환경을 먼저 준비하고 `pip install -e '.[viz,serve,validation,dev]'`로 설치합니다. 모델은 로컬 Hugging Face 캐시에서만 읽으며, 실행 중 모델을 자동 다운로드하지 않습니다. 현재 지원하는 decoder는 Llama와 Qwen3입니다.

### 실제 LLM 비교

```bash
# GPU와 모델의 정확도 검증 후 실행하는 작은 전체 모델 실험
.venv/bin/python -m kernelscope.cli serve run \
  --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/tiny.yaml \
  --policy heuristic --policy fixed:8 --kv-gib 1 \
  --repeats 3 --warmup-runs 1 --warmup-steps 2 \
  --out ../kernelscope/results/serve_4090/tiny

# 전체 캠페인: 같은 길이, 긴·짧은 요청 혼합, 실행 중 추가 요청
bash scripts/campaign.sh

# 후속 검증: 두 모델 × 자연어 조건 3개 × 시드 2개 × 정책 3개 × 반복 3회
.venv/bin/python scripts/followup_campaign.py \
  --out ../kernelscope/results/serve_4090/followup_new --plan-only
# --plan-only를 빼면 실제 측정합니다.
# 기본값은 각 정책으로 전체 시나리오를 워밍업하며, 측정마다 정책 캐시는 비웁니다.

# GPU 없이 실행 로직을 검증하는 별도 CPU 데모
.venv/bin/python -m kernelscope.cli serve demo --out results/serve/cpu_demo
```

```bash
# 연산 클래스 분해 진단(대조 실행 + 계측 실행, 성능 주장 아님)
.venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario scenarios/graduation_ragged.yaml --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
  --kv-gib 10 --out ../kernelscope/results/serve_4090/diagnose_new/ragged
.venv/bin/python -m kernelscope.cli serve diagnose-report demo_data/serve_4090/diagnose_20260927/ragged   # GPU 없이 재생성
```

decode step을 8개 연산 클래스로 나누어 CUDA event로 시간을 재고, 각 클래스를 DRAM/텐서코어 상한(측정 기반, θ=0.7)에 대어 memory_bound/compute_bound/launch_bound/parallelism_candidate/below_ceiling_unknown으로 판정합니다. 바이트/FLOP 비용 모델은 필수 트래픽만 세는 하한 추정이라 pct_dram·pct_tc도 하한 추정입니다. 결과와 재현 방법은 [연산 클래스 분해](docs/experiments/2026-09-27-op-breakdown.md)에 있습니다.

성능 실험은 전체 사전학습 모델을 사용합니다. `graduation_*`은 합성 토큰 입력, `heldout_text_*`은 직접 작성한 자연어 문단을 길이에 맞춰 구성한 입력입니다. attention 시간, 전체 decode 시간, CPU 선택 비용, 요청별 TPOT, 반복별 결과와 생성 토큰 일치를 따로 기록합니다. `serve demo`의 작은 랜덤 CPU 모델은 실행 로직 확인용이며 GPU 성능 근거가 아닙니다. 기존 커널 실측으로 만든 선택표는 `demo_data/dispatch_paged_cold.csv`에 있습니다.

모델 선택의 CPU 시뮬레이터는 로컬 C 컴파일러가 있으면 빠른 실행 경로를 사용하고, 없으면 NumPy 구현으로 돌아갑니다. 컴파일·로드 비용과 실행 경로는 별도로 기록하며 `KERNELSCOPE_SIMULATOR=python`으로 기준 구현을 강제할 수 있습니다. `.cache/`의 생성 라이브러리는 배포하지 않고 C 소스를 패키지에 포함합니다.

생성 토큰이 달라지면 실험 결과를 보존하고 검증 실패를 반환합니다. `campaign.sh`는 완료된 시나리오를 보존하며 다른 시나리오를 계속 실행하고, 마지막에 검증 실패가 하나라도 있으면 종료 코드 1을 반환합니다. 같은 결과 경로로 재실행하면 완료된 시나리오는 건너뜁니다. 부정적인 결과도 삭제하거나 성공으로 바꾸지 않습니다.

```bash
# 실제 모델의 자연어 이어쓰기. 단일 요청 데모이며 성능 비교 실험은 아닙니다.
.venv/bin/python -m kernelscope.cli serve generate \
  --prompt '인공지능 모델에서 GPU 커널의 역할은' --max-new-tokens 48 --policy heuristic

# 같은 입력의 수치적 출력 검증과 결과 보고서/발표용 그림
.venv/bin/python scripts/verify_qwen.py --out docs/experiments/qwen3-4b-validation.json
.venv/bin/python scripts/summarize_campaign.py <캠페인 폴더> --out docs/experiments/serving-results.md
.venv/bin/python scripts/plot_campaign.py <캠페인 폴더> --out docs/img/serving_comparison.png
```

시연은 `./run.sh`로 합니다(인자 없이 실행하면 사용법). GPU 없는 장비에서는 `check`·`dashboard`·`verify`·`traffic`(공개 트레이스 재생)·`defaults`(두 라이브러리 기본값 손실 표), 연구실 4090에서는 `kernel`·`serve`·`diagnose`·`divergence`·`generate`·`gpu-all`이 라이브 측정을 돌리고 결과를 `$KERNELSCOPE_RESULTS/demo_runs/<시각>/`에 남깁니다. 다른 GPU 프로세스가 있으면 기다리지 않고 이유를 출력한 뒤 멈춥니다.

```bash
./run.sh check            # 환경·번들·GPU 점검
./run.sh dashboard        # 저장된 실측 대시보드 (GPU 불필요)
./run.sh gpu-all          # 최악 셀 커널 비교 → 실생성 비교 → 연산 분해 → 분기 사건 분류 (약 4분)
./run.sh generate "GPU 커널 선택이 중요한 이유는" hybrid
./run.sh verify --tests   # 문서 수치 재계산 + CPU 테스트
./run.sh traffic          # Azure 트레이스를 연속 배치로 재생: 분할 휴리스틱이 손해 보는 step 비율 (GPU 불필요, 약 1분)
```

```bash
make test           # CPU 테스트
make test-gpu       # 유휴 GPU에서 CUDA 검증
make verify         # GPU 없이 문서의 주요 수치를 커밋된 원본 데이터에서 다시 계산해 대조
make figures        # 보고서 그림을 demo_data에서 직접 생성 (docs/report/fig)
make package-demo  # 원본 측정 결과를 해시와 함께 휴대 가능한 데모로 복사
```

### GPU 없는 장비에서 확인하기

CUDA가 없는 노트북에서도 CPU용 PyTorch로 다음을 확인할 수 있습니다.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pandas pyarrow pyyaml pytest matplotlib 'streamlit>=1.64' 'plotly>=6' safetensors==0.8.0 tokenizers==0.22.2
.venv/bin/pip install -e . --no-deps
make test     # CPU 테스트 (GPU 테스트는 제외, flash-attn/transformers 없는 항목은 skip)
make verify   # 531개 수치 재현 검사: 커널 실측, 실제 생성, 후속 실험, 선택 비용, 성능 모델·혼합 정책 검증, 일관성, GPGPU-Sim 스윕, 연산 분해, 혼합 정책 실생성, 분기 사건 분류, 트레이스 재생, 두 라이브러리 기본값, 64K, 커널 탐침, FlashInfer 엔진 캠페인, 라이브러리 무관 선택표
make demo     # 저장된 실측 결과로 대시보드 실행
```

GPU 없이 기록된 측정으로 돌리는 분석 두 가지도 있습니다. `python -m scripts.evaluate_hybrid`는 성능 모델·측정 테이블·혼합 정책의 선택 손실을 leave-one-out으로 비교하고([결과](docs/experiments/2026-09-26-hybrid-policy.md)), `python -m scripts.classify_divergence --campaign <캠페인 폴더>`는 새 캠페인의 생성 토큰 불일치를 분기 사건으로 세고 교사 강제 진단(`scripts/check_policy_numerics.py`)으로 `tie_1ulp/tie_2ulp/clear`를 분류합니다([결과](docs/experiments/2026-10-02-hybrid-validation.md)); `python -m scripts.analyze_mismatches`는 9월 22일 후속 캠페인의 불일치를 같은 방식으로 정리합니다([결과](docs/experiments/2026-09-26-mismatch-analysis.md)). 혼합 정책은 `--policy hybrid:<table.csv>:0.2`로 서빙 실험에 쓸 수 있고, `--policy flashinfer_cudacore`(또는 `flashinfer`)는 같은 규약에서 FlashInfer 커널을 돌려 비교합니다(`kernelscope/serve/attention.py`; plan() 비용은 `policy_us`에 기록). `--policy table_any:demo_data/dispatch_paged_cold_any.csv[:<plan_us>]`는 FA2 분할과 FlashInfer 두 변형을 모두 후보로 둔 라이브러리 무관 선택표로, step마다 분할 수와 attention 백엔드를 함께 고릅니다(`plan_us`를 주면 FlashInfer 백엔드에 step당 plan() 비용을 더해 비교; 선택한 백엔드는 `steps.parquet`의 `attention` 열에 기록). `python -m scripts.traffic_replay --trace <Azure/BurstGPT CSV> --rate-scale 1 --out <폴더>`는 공개 요청 트레이스를 연속 배치로 재생해 분할 휴리스틱이 손해 보는 decode step의 비율을 성능 모델로 셉니다([결과](docs/experiments/2026-10-06-problem-scope.md)).

`kernelscope verify`는 `demo_data/`와 `docs/experiments/`의 RTX 4090 원본 기록만 읽습니다. 각 수치를 프로젝트의 분석 코드로 다시 계산하고, 해당 문서가 여전히 그 값을 적고 있는지도 확인합니다. 하나라도 어긋나면 종료 코드 1을 반환합니다. `--only kernel,serve`로 일부 묶음만, `--data <결과 폴더>`로 원본 결과 폴더를 지정할 수 있습니다. `serve demo`의 CPU 실행은 스케줄링과 토큰 일치 로직만 확인하며, CPU SDPA는 `num_splits`를 사용하지 않으므로 커널 선택의 성능 효과를 재현하지 않습니다.

이 README의 아래 예시는 개발 장비의 절대 경로를 사용합니다. 다른 장비에서는 다음 환경 변수로 바꿉니다.

| 변수 | 의미 | 기본값 |
|---|---|---|
| `KERNELSCOPE_PYTHON` | `scripts/demo.sh`가 사용할 Python | `.venv/bin/python` |
| `KERNELSCOPE_RESULTS` | 대시보드가 읽을 결과 폴더 | `../kernelscope/results`, 없으면 `demo_data/` |
| `ACCELSIM_ROOT` | 빌드된 Accel-Sim 경로 | `/home/skkai/accelsim/accel-sim-framework` |
| `KERNELSCOPE_SIMULATOR` | 성능 모델 시뮬레이터 구현 (`python`이면 NumPy 기준 구현) | C 컴파일러가 있으면 C 구현 |

## 커널 측정 기반

Dual-track evaluation harness for GPU inference kernels: plug in an **existing**
attention kernel (FlashAttention-2, FlashDecoding, SDPA backends,
or your own CUDA/Triton attention kernel) and get

1. **real-hardware** numbers on CUDA GPUs (current campaign: RTX 4090; earlier data: A100) — **without Nsight**: CUDA-event latency,
   torch.profiler kernel time + launch geometry (grid/block/registers/smem →
   occupancy estimate), and an analytic traffic/FLOP model that turns kernel time
   into achieved GB/s and TFLOPS against *measured* ceilings; and
2. **simulated** numbers from Accel-Sim's `SM80_A100` and experimental `SM89_RTX4090` models plus *what-if*
   variants (L2 size, HBM bandwidth, SM count) that no measurement can give, with
   the instruction mix read straight from the NVBit trace,

joined on one workload key so the two tracks validate each other
(sim cycles vs measured kernel time) and the what-if sensitivity gets a verdict
("bandwidth-bound", "parallelism-bound", "insensitive").

The current SM89 simulation timing is **not a calibrated RTX 4090 performance predictor**.
The serving results above come from the real GPU; dashboard hardware what-if values come
from the separately fitted surrogate and remain predictions.

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
PY=/home/skkai/miniforge3/envs/gradkernel/bin/python      # always absolute paths on this box
$PY -m kernelscope.cli list                                # registered plugins
$PY -m kernelscope.cli ceilings --out results/machine_ceilings.json           # HBM / fp16-GEMM peaks
CUDA_VISIBLE_DEVICES=0 $PY -m kernelscope.cli sweep --grid grids/w1_min.yaml \
    --plugins fa2,flashdecoding,sdpa_flash,sdpa_efficient --results results/hw \
    --ceilings results/machine_ceilings.json --python $PY
CUDA_VISIBLE_DEVICES=0 $PY -m kernelscope.cli simsweep --grid grids/sim_smoke.yaml \
    --plugins fa2,flashdecoding --results results/sim --variants base,bw_x2,bw_half,l2_x2,l2_half,sm_x2,sm_half \
    --work-dir /home/skkai/accelsim/kernelscope_sim --device-index 0 --python $PY
$PY -m kernelscope.cli report --results results/hw results/sim --out results/summary.csv
$PY -m kernelscope.cli plot --results results/hw --ceilings results/machine_ceilings.json --out results/roofline.png

# machine spec used by the surrogate performance model (dram/L2 bandwidth, L2 hit curve, block→SM placement)
$PY -m kernelscope.cli machine --out machines/rtx4090.json

# bench: in-process real-HW batch (no subprocess/ncu per cell) over one or more grids, both cache states in one pass
$PY -m kernelscope.cli bench --grid grids/dispatch_s1.yaml --plugins fa2,flashdecoding,fd_s8,fd_s16 \
    --results results/hw_4090/uniform_s1_dense --cache-state cold,warm
    # --cache-state: 'cold' writes 4 x L2 bytes with a dedicated flush kernel before every timed
    # call (serving-realistic: another layer's weights stream through L2 between two attention
    # calls of the same layer); 'warm' only inserts a one-float iteration-marker kernel. A comma
    # list runs one pass per state and both land in the same summaries.jsonl / parquet with a
    # `cache_state` extra column. An interrupted or timed-out run is safe to re-run:
    # cells already recorded as `ok` are skipped (resume is automatic, `--no-resume` disables it).

# dispatch-table: best interchangeable variant per workload + the library heuristic's regret against it
$PY -m kernelscope.cli dispatch-table --results results/hw_4090/uniform_s1_dense --family dense \
    --cache-state cold --out results/hw_4090/tables/uniform_s1_dense_cold.csv
    # --family dense|paged selects which kernel variants are interchangeable (fa2/flashdecoding/fd_s{N}
    # vs their _paged counterparts); prints a regret_summary (median/max/worst-key) plus the table head.
```

Measure on an **idle** GPU: check `nvidia-smi` first (the sweep records utilisation and
foreign processes on the target GPU at start and warns). Long-running jobs of your own on
another GPU are fine; on the same GPU they inflate kernel times several-fold.

`simsweep` needs the built Accel-Sim tree (default `/home/skkai/accelsim/accel-sim-framework`,
override with `--accelsim-root` or `ACCELSIM_ROOT`); see [docs/setup/accelsim_4090_gate_report.md](docs/setup/accelsim_4090_gate_report.md).

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
    paged.py             paged-KV-cache block_table helpers backing the builtin plugins' _paged variants
  backends/
    realhw/              latency.py (CUDA events), kprofile.py (torch.profiler + occupancy),
                         ncu.py (optional), sweep.py (orchestrator)
    realhw/cache.py      IterationHooks: cold-flush / warm-marker boundary kernels before each timed call
    realhw/batch.py      in-process batch runner used by `bench` (no subprocess/ncu per cell)
    accelsim/            paths / config (what-if variants) / trace / stats / sweep
  analysis/
    trace_mix.py         opcode histogram + global bytes from the raw NVBit trace
    report.py            one wide row per (kernel, workload) across tracks + what-if verdict
    dispatch.py           dispatch_table / regret_summary: best interchangeable variant per
                          workload (dense or paged family) and the library heuristic's regret
  bench/
    ceilings.py            measured HBM and fp16-GEMM peaks (torch only)
    machine.py             measure_machine(): full MachineSpec (dram/L2/CTA bandwidths, L2 hit
                           curve, block→SM placement) written to machines/<gpu>.json
    stream.py              Triton streaming-read kernel for DRAM/L2 bandwidth measurements
    cuda_ext.py            small CUDA kernels Triton can't express: SM blocker + %smid probe
    sm_blocker.py          real-hardware SM-count what-if: occupies n SMs while a target kernel runs
    placement.py           records %smid for a grid filling every SM (block-to-SM placement rule)
  results/store.py       long-format parquet, one file per write (`cache_state` extra column
                         when written by `bench`: 'cold' or 'warm')
grids/                   workload grids (w1_min, decode_full, prefill, sim_smoke, dispatch_s{1,2},
                        ragged_s{1,2} — uniform vs. ragged-batch decode shapes for dispatch-table)
machines/                measured MachineSpec JSON per GPU (`kernelscope machine --out`)
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

Run the whole suite with a single `pytest -q` (don't split it into a GPU pass and a CPU pass).
GPU tests are ordered first because torch.profiler can stop recording CUDA kernels after a pause
in a process that already profiled (see `tests/conftest.py`).
