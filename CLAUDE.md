# kernelscope — 학사 졸업작품 (이선재, 2026-2학기)

GPU 프로파일링·시뮬레이션 기반 LLM 추론 attention 커널 동적 선택 시스템. 작업 브랜치 `design-1-3`. **연구급 아님 — 코어 스코프만으로 완성.**

## 문서 인덱스 (필요할 때만 읽기)

- @PLAN.md — 계획 v3(단일 원본): 한 줄 정의, 확보한 것, 남은 작업 순서, 중단 조건.
- README.md → docs/graduation.md → docs/STATUS.md — 실행법, 연구 질문, 진행 기록(최신 항목이 위).
- docs/plan/ — 개별 작업 계획, docs/handoff/ — 호스트 간 인계 기록.

## 호스트 구분

어느 호스트인지 모르면 `nvidia-smi`로 확인한다.

- **로컬 WSL2** (`/root/sunjae/gpu_graduation/kernelscope`): GPU 없음, RAM 3.8GiB. GPU가 필요한 실험은 설계·도구까지만 만들고 "미실행"으로 표기. 드라이버 설치 금지. pytest/pip/streamlit/빌드는 한 번에 하나, 서브에이전트 동시 3개 이하.
- **연구실 RTX 4090** (`/home/skkai/...`, 경로는 README 참고): PLAN §2의 GPU 필요 작업을 실행하는 곳. 파이썬은 README의 절대 경로를 쓴다.

## 불변 규칙

- 시뮬레이터 리포(`gpgpu-sim_distribution/`, `accel-sim-framework/`)는 upstream 클론 — 직접 수정 금지. 로컬 GPGPU-Sim 실행은 `experiments/gpgpusim/run_sim.sh`(README에 CUDA 11.8 우회 규칙). 원시 결과는 리포 밖에 두고, 문서에 인용하는 표만 `docs/experiments/`에 복사해 커밋한다.
- 수치는 문서에 쓰기 전에 `kernelscope verify`에 검사 항목을 추가한다. 불일치·실패 결과도 삭제하지 않는다.
- 계획이 바뀌면 PLAN.md를 고친다. 다른 호스트는 커밋·푸시된 내용만 보므로, 고친 뒤 커밋이 필요하다고 사용자에게 알린다.
- git 커밋/푸시는 사용자가 명시적으로 요청할 때만.
