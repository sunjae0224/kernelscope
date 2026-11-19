최종보고서 산출물 안내

작성일: 2026-10-28 (Asia/Seoul)
기준 브랜치: design-1-3 / 4cdd4fd

완성본: 연구논문작품_최종보고서_이선재.docx, 동일 이름의 PDF
분량: 표지 1쪽 + 본문 31쪽 = 32쪽
구성: 요약, 서론, 관련연구, 제안 작품 소개, 구현 및 결과분석, 결론 및 소감, 참고문헌, 코드 부록
그림 5개와 표 19개. 평가표와 서명은 작성하지 않음.

사용자의 midterm_report/연구논문작품_최종보고서.pdf에는 완성본을 배치하고,
원래의 4쪽 학교 양식은 같은 폴더의 연구논문작품_최종보고서_원본양식.pdf에 보존함.
동일 폴더에 편집용 연구논문작품_최종보고서.docx도 배치함.

본문 작성 소스
src/content.py → src/content.json → src/build.js → DOCX
src/figures.py → figures/의 직접 제작 PNG·SVG
의존성: Python3, matplotlib, docx npm9.7.2, 한글 맑은고딕 글꼴, LibreOffice
저장소 루트에서:
  npm ci --prefix docs/report/final/src --ignore-scripts
  python3 docs/report/final/src/figures.py
  python3 docs/report/final/src/content.py
  node docs/report/final/src/build.js
  libreoffice --headless --convert-to pdf --outdir docs/report/final \
    docs/report/final/연구논문작품_최종보고서_이선재.docx

근거와 검증
- evidence/results.json: 원자료 CSV/JSONL의 독립 집계와 조건·검증 키
- evidence/style_and_references.json, references.json: 중간보고서 양식·검증한 문헌
- evidence/changes_from_midterm.json: 중간보고서 대비 변경 사항
- evidence/simulator_audit.txt, results_audit.txt: 구현·결과의 세부 감사 메모
- evidence/verify.csv: 기존 재현 검증944개 + 최종보고서 파생수치4개, 총948개PASS
- evidence/provenance.json: 브랜치 커밋과 입력PDF의 원본 해시
- evidence/layout_validation.json: PDF페이지·그림·쪽번호·인용·배치 검증

다시 수치 검증:
  .venv/bin/python -m kernelscope.cli verify
최종보고서에 추가한 파생 감소율·처리량 검증만 실행:
  .venv/bin/python -m kernelscope.cli verify --only finalreport

보고서 작성 중 GPU 실험은 새로 수행하지 않았으며 저장소의 기존 측정을 재계산함.
시뮬레이터의 계산 비용과 예측 정확도, 커널 시간과TPOT,
실측과 모델 예측, 토큰동일성과 정답률을 구분하여 작성함.
commit/push는 수행하지 않음.
