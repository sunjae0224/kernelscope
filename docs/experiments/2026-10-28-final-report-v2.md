# 최종보고서 v2의 공란 수치 (2026-10-28)

`docs/report/final_v2/남은_실험.md` 1장의 공란 10칸을 채운 근거다. 모두 GPU 없이 계산했고, 연구실 RTX 4090 호스트에서 `CUDA_VISIBLE_DEVICES=`로 GPU를 숨기고 실행했다(같은 시각 GPU는 다른 사용자의 학습 작업이 쓰고 있었다). 2장의 선택 실험 가운데 GPU가 필요 없는 (d)·(e)·(f)는 `2026-10-24-final-evidence.md` §4·§5·§3에 있다. GPU가 필요한 (a)·(b)·(c)는 하지 않았다.

## 1. Accel-Sim 6개 조건의 성능 시뮬레이터 예측 (표 10, 공란 3~9)

실측과 Accel-Sim 값은 `docs/setup/accelsim_4090_gate_report.md`의 표와 같은 원본(`../kernelscope/results/sim_4090/20260918-complete/validation.csv`)을 `demo_data/accelsim/validation_20260918.csv`로 옮겨 쓴다. 실측은 torch.profiler의 커널 시간이고 워밍업 뒤에 쟀다(게이트 보고서: "Real-HW measurement follows warm-up"). 그래서 성능 시뮬레이터도 warm 상태로 예측했다(`kernelscope.model.predict.predict(kernel, workload, machines/rtx4090.json, models/rtx4090.json, "warm")`). 헤드 구성은 질의 32·KV 8, d = 128, fp16이다.

| 조건 | 실측 µs | Accel-Sim µs | Accel-Sim/실측 | 성능 시뮬레이터 µs (warm) | 시뮬레이터 오차 | 보정 격자 |
|---|---:|---:|---:|---:|---:|---|
| FA2 B=1, L=1K | 63.584 | 67.5 | 1.061 | 61.0 | 4.1% | 밖(보간) |
| FA2 B=1, L=8K | 494.336 | 515.9 | 1.044 | 460.1 | 6.9% | 안 |
| FA2 B=16, L=1K | 64.303 | 109.7 | 1.707 | 61.0 | 5.2% | 밖(보간) |
| FlashDecoding B=1, L=1K | 10.016 | 15.5 | 1.548 | 10.8 | 8.0% | 밖(보간) |
| FlashDecoding B=1, L=8K | 27.456 | 73.6 | 2.680 | 23.3 | 15.0% | 안 |
| FlashDecoding B=16, L=1K | 38.2235 | 116.8 | 3.056 | 37.2 | 2.7% | 밖(보간) |

- 성능 시뮬레이터의 오차는 2.7~15.0%(중앙값 6.1%)다. 같은 조건에서 Accel-Sim은 실측의 1.04~3.06배다.
- **캐시 상태.** cold로 예측하면 오차가 15.9~140.2%로 커진다(FlashDecoding 세 조건이 106~140%). 표 10에는 "warm 기준"을 적어야 한다.
- **보정 격자 여부.** 보정 격자는 균일 길이·KV 헤드 8에서 B ∈ {1, 4, 16, 64} × L ∈ {512, 2048, 8192, 32768}이다(`kernelscope/model/validate.py`의 `TRAIN_B`, `TRAIN_L`). 따라서 L = 8K의 두 조건(FA2, FlashDecoding, 모두 B = 1)은 격자 안의 조건이고, L = 1K의 네 조건은 보정에 쓰지 않은 보간 조건이다. 단, 보정에 쓴 측정(2026-09-22 P0 캠페인)과 이 표의 실측(2026-09-18 Accel-Sim 게이트)은 서로 다른 실행이다. `남은_실험.md` 1.2절은 "FA2 B=1·L=8K만 격자 안, 나머지 L=1K 다섯 칸"이라고 적었지만, L = 1K는 네 칸이고 FlashDecoding B=1·L=8K도 격자 안이다.
- **해석 범위.** 성능 시뮬레이터는 같은 GPU의 실측으로 보정했고, Accel-Sim은 보정하지 않은 사이클 수준 시뮬레이터다. 따라서 "Accel-Sim보다 정확하다"가 아니라 "실측으로 보정한 시뮬레이터는 같은 조건에서 15% 이내"로 쓴다.

## 2. GPGPU-Sim 축소 커널 1회 실행 시간 (표 7, 공란 2)

GPGPU-Sim 실행 로그(`sim.log`의 `gpgpu_simulation_time`)는 로컬 WSL2 노트북의 `gpu_graduation/results/gpgpusim/<실행>/sim.log`에만 있다. 이 호스트에는 없어서 다시 확인하지 못했다. 아래 값은 사용자가 2026-10-28 `남은_실험.md` 1.1절에 옮긴 값을 `docs/experiments/gpgpusim-splitkv-sweep.csv`의 `sim_wall_s` 열로 추가한 것이다. 사이클 수와 같은 행이다. 균일 길이 분할 2·4·8의 실행 시간은 기록에 없어 비워 두었다.

| workload | S | 실행 시간(s) | gpu_sim_cycle |
|---|---:|---:|---:|
| ragged | 1 | 330 | 452,140 |
| ragged | 2 | 230 | 230,648 |
| ragged | 4 | 226 | 121,071 |
| ragged | 8 | 288 | 86,491 |
| ragged | 16 | 362 | 62,566 |
| uniform | 1 | 219 | 53,004 |
| uniform | 16 | 351 | 26,102 |

보고서 표 7의 대상(혼합 길이 2048+128×15, 분할 1)의 1회 실행은 330초다. 실행 시간은 사이클 수에 비례하지 않는다(분할 16은 사이클이 7.2배 적지만 더 오래 걸린다). 시뮬레이터 실행 시간은 시뮬레이션하는 명령어 수와 CTA 수에도 좌우되기 때문이다. 로컬에서 `grep -h 'gpgpu_simulation_time' results/gpgpusim/ragged_s1/sim.log`로 원 로그와 맞는지 한 번 확인하면 된다.

## 3. 단위 테스트 수와 verify 소요 시간 (공란 1, 10)

이 두 값은 실행 결과라서 verify 항목이 없다. 출력 그대로 적는다.

- 단위 테스트: `CUDA_VISIBLE_DEVICES= nice -n 19 make test`(= `pytest -q -p no:cacheprovider -m 'not gpu'`) → **723 passed, 5 skipped, 15 deselected**(deselected는 `gpu` 표시 테스트). 이 수에는 이번에 추가한 테스트 5개(Mooncake 로더 1, 분할 없음 임계 4)가 포함된다.
- verify: `CUDA_VISIBLE_DEVICES= /usr/bin/time -p make verify` → **PASS 1,109 / 1,109, real 31.21 s**(두 번 더 재면 31.37 s, 31.28 s). 항목 수는 944개에서 이번에 추가한 165개(`final2.*` 34, `delta.*` 39, `threshold.*` 36, `mooncake.*` 56)를 더한 1,109개다.
- 보고서는 두 값을 "GPU가 없는 노트북 환경"의 값으로 적고 있다. 이 값은 연구실 호스트(32 CPU)에서 GPU를 숨기고 잰 값이므로, 노트북에서 다시 재면 시간은 달라질 수 있다. 보고서 문장은 "GPU 없이"로 고쳤다.
- 로컬 WSL2에만 있는 `finalreport2` verify 9항목은 이 수에 들어 있지 않다. 그 항목을 합쳐 커밋하면 보고서의 항목 수(1,109)도 함께 고친다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -c "
from pathlib import Path
from kernelscope.verify import _accelsim_surrogate
for c, v in _accelsim_surrogate(Path('.').resolve(), Path('demo_data').resolve()).items(): print(c, round(v['us'], 3), round(v['err_pct'], 2))"
CUDA_VISIBLE_DEVICES= nice -n 19 make test
CUDA_VISIBLE_DEVICES= /usr/bin/time -p make verify      # final2.* 항목
```
