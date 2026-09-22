# Qwen3-4B 실제 생성 비교

RTX 4090의 단일 프로세스 전체 모델 decode 실험. 합성 토큰 프롬프트로 길이를 통제했다. 네트워크나 운영 서버 부하를 측정한 결과는 아니다.

TPOT는 요청별 생성 토큰 간 간격의 평균이다. 첫 생성 토큰 뒤에 다른 요청의 prefill이 실행되면 그 대기도 포함한다. Decode wall은 정책 CPU 계산과 토큰 추출을 포함한 각 decode 호출의 host 관측 시간이며 prompt admission은 제외한다. 이벤트 기반 step 시간과 구분한다.

## arrivals

모델: `Qwen/Qwen3-4B-Instruct-2507` · 반복: 5 · 요청 수: 48 · seed: 0

| 정책 | Attention ms/step | Decode wall ms/step | 선택 µs/step | TPOT ms | TPOT 배속 | 95% 반복 bootstrap | 토큰 일치 |
|---|---:|---:|---:|---:|---:|---|---|
| fixed8 | 4.981 | 19.088 | 2.8 | 50.048 | 1.181× | 1.176–1.188 | 일치 |
| heuristic | 10.609 | 24.641 | 3.0 | 59.094 | 1.000× | 1.000–1.000 | 일치 |
| model | 5.001 | 56.224 | 37027.4 | 107.867 | 0.548× | 0.546–0.550 | 일치 |
| table | 5.044 | 19.229 | 11.6 | 50.340 | 1.174× | 1.171–1.177 | 일치 |

배속은 휴리스틱 TPOT / 후보 TPOT이다. 소수 반복의 bootstrap 구간은 이 실행 안의 변동을 설명하며 다른 GPU·날짜·실제 요청 분포에 대한 일반화 보장은 아니다. 토큰 불일치가 있는 정책의 속도 차이는 출력 보존 개선으로 주장하지 않는다.

원본: `/home/skkai/AI_Accelerator/kernelscope/results/serve_4090/graduation_20260922/arrivals`

## ragged

모델: `Qwen/Qwen3-4B-Instruct-2507` · 반복: 5 · 요청 수: 32 · seed: 0

| 정책 | Attention ms/step | Decode wall ms/step | 선택 µs/step | TPOT ms | TPOT 배속 | 95% 반복 bootstrap | 토큰 일치 |
|---|---:|---:|---:|---:|---:|---|---|
| fixed8 | 9.139 | 23.752 | 3.4 | 33.385 | 1.829× | 1.826–1.832 | 일치 |
| heuristic | 36.873 | 51.359 | 3.6 | 61.058 | 1.000× | 1.000–1.000 | 일치 |
| model | 9.132 | 35.627 | 11899.1 | 45.329 | 1.347× | 1.341–1.353 | 일치 |
| table | 9.238 | 24.082 | 7.7 | 33.801 | 1.806× | 1.805–1.808 | 일치 |

배속은 휴리스틱 TPOT / 후보 TPOT이다. 소수 반복의 bootstrap 구간은 이 실행 안의 변동을 설명하며 다른 GPU·날짜·실제 요청 분포에 대한 일반화 보장은 아니다. 토큰 불일치가 있는 정책의 속도 차이는 출력 보존 개선으로 주장하지 않는다.

원본: `/home/skkai/AI_Accelerator/kernelscope/results/serve_4090/graduation_20260922/ragged`

## uniform

모델: `Qwen/Qwen3-4B-Instruct-2507` · 반복: 5 · 요청 수: 32 · seed: 0

| 정책 | Attention ms/step | Decode wall ms/step | 선택 µs/step | TPOT ms | TPOT 배속 | 95% 반복 bootstrap | 토큰 일치 |
|---|---:|---:|---:|---:|---:|---|---|
| fixed8 | 3.603 | 18.286 | 3.4 | 27.619 | 0.993× | 0.992–0.994 | 불일치/미검증 |
| heuristic | 3.585 | 18.129 | 3.2 | 27.424 | 1.000× | 1.000–1.000 | 일치 |
| model | 3.590 | 33.062 | 14892.2 | 42.428 | 0.646× | 0.645–0.647 | 일치 |
| table | 3.585 | 18.147 | 8.0 | 27.495 | 0.997× | 0.995–1.000 | 일치 |

배속은 휴리스틱 TPOT / 후보 TPOT이다. 소수 반복의 bootstrap 구간은 이 실행 안의 변동을 설명하며 다른 GPU·날짜·실제 요청 분포에 대한 일반화 보장은 아니다. 토큰 불일치가 있는 정책의 속도 차이는 출력 보존 개선으로 주장하지 않는다.

원본: `/home/skkai/AI_Accelerator/kernelscope/results/serve_4090/graduation_20260922/uniform`

## 재현

```bash
REPEATS=5 bash scripts/campaign.sh <새 결과 폴더>
.venv/bin/python scripts/summarize_campaign.py <결과 폴더> --out docs/experiments/2026-09-22-serving-results.md
```

각 manifest에는 모델 snapshot, 시나리오 해시, 패키지 버전, 정책 입력 해시, 반복 순서와 GPU 사용 상태가 기록된다. 결과는 기존 디렉터리를 덮어쓰지 않는다.
