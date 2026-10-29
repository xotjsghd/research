# medical-lina 전체 구현 계획

목표: 의료 이미지 sampling 개선. LINA는 참고 방법이며 의료용 비교 조건은 실험으로 결정한다.
아래 미래 파일은 계획일 뿐 아직 생성하지 않았다. 초기에 불필요한 빈 구현을 만들지 않는다.

| 단계 | 파일(예정 포함) | 책임 / 입출력 | 상태 |
|---|---|---|---|
| 환경 | check_environment.py | Python·패키지·GPU 상태 출력 | 생성 |
| 기본 생성 | generate_cxr.py | 단일/CSV 프롬프트 × seed → PNG·설정·manifest | 생성 |
| 프롬프트 | prompts/01_*.csv ~ 07_*.csv | 실험용 프롬프트 세트 01–07 (계획에 있던 smoke_test.csv는 만들지 않음) | 생성 |
| 기준 평가 | eval_*_medgemma.py | 세트별 이미지 폴더 → MedGemma 판정 responses.jsonl (03–06 완료, 07 정식 평가 전) | 생성 |
| 데이터 조사 | data_analysis/summarize_dataset.py | 허가된 로컬 데이터의 소견·위치 등 분포 요약 | 이후 |
| 의료 규칙 | rules/build_rules.py, rules/clinical_rules.json | 후보+근거+검토 → 적용 조건·예외 포함 R | 이후 |
| 문제 구성 | prompts/build_prompts.py | 선정한 조건·템플릿 → 프롬프트·기대 조건 | 이후 |
| 생성 공통화 | models/sana_backend.py | 현재 생성 코드의 모델 로딩을 평가·탐색에서 재사용 | 이후 |
| 평가 | evaluation/findings.py | 선정한 의료 모델 → 소견 근거·불확실성 | 이후 |
| 기하 측정 | evaluation/geometry.py | CXAS 마스크 → 측정치; 관련 과제일 때 사용 | 이후 |
| 목표 비교 | evaluation/compare.py | 관측/기대 조건 → 충족·불일치·판정 보류 | 이후 |
| 평가 실행 | evaluate_cxr.py | manifest → 평가 결과; 기하 사용 여부는 과제에 따름 | 이후 |
| 오류 조사 | analysis/summarize_failures.py | 반복 오류 유형·빈도 정리; 데이터 인과로 단정 금지 | 이후 |
| 비교 조건 | intervention/build_reference.py | X+선정한 오류 정책 → X_reference(예: X_biased) | 이후 |
| 토큰 개입 | intervention/token_edit.py | 선택 토큰·gamma1 → c_prime; 처음에는 비활성 가능 | 이후 |
| 샘플링 개입 | intervention/sana_guidance.py | 동일 latent/t에서 조건별 예측 조합 → scheduler 업데이트 | 이후 |
| 검색 | search_gamma.py | 고정 비교 정책+프롬프트/seed → gamma 후보 생성·평가 | 이후 |
| 학습셋 | aim/build_labels.py | 탐색 결과 → 프롬프트별 라벨·train/val/test 분리 | 이후 |
| AIM | aim/model.py, train_aim.py | 텍스트 encoder+MLP → 개입 강도 회귀 | 이후 |
| 온라인 생성 | generate_with_aim.py | X → gamma·비교 조건 → 의료 이미지 | 이후 |
| 비교 실험 | experiments/compare_methods.py | CFG·생략 조건·오류 조건·AIM 비교 | 이후 |

## 평가 결과 폴더 버전 (세트 06)

- 저장소의 `evaluations/finding_independence_medgemma_v1/`이 README 수치의 근거인 실제 결과다.
- `eval_independence_medgemma.py`의 `--out-dir` 기본값은 `..._v2`다. 기존 결과가 있으면 스크립트가 멈추므로(`--resume` 제외), 다시 실행하면 v1은 그대로 두고 v2 폴더에 새로 저장된다.
- v1 기록의 질문 문구, 모델 commit, `max_new_tokens`(320)는 현재 코드와 같다. v2 결과는 저장소에 없다.

## 설계상 고정하지 않은 것

- 평가 모델(MedGemma/MAIRA-2 등), 임상 규칙, c_biased 구성법, 감마 탐색 방식은 아직 미확정.
- 데이터의 공기 빈도를 임상적 필수/인과 규칙으로 취급하지 않는다.
- prompt-only AIM은 비교 조건 선택 정책이 고정된 경우부터 시작. 같은 X에 여러 비교 조건을 쓰면 AIM 입력 재검토.
- gamma 라벨은 개입 구현·모델·스케줄러에 종속된다. 바뀌면 재탐색 검토.
- 프롬프트/환자/템플릿 등 목적에 맞는 단위로 분리하고 test는 오류 정책·탐색에 사용하지 않는다.
- Phase 2는 매 step 예측·조합·업데이트를 반복. 인코딩/AIM 예측은 원칙적으로 최초 한 번.
- 기본 생성 → 오류 확인 → 비교 조건 효과 검증 → 감마 탐색 → AIM 순서.
- 별도 generate_batch.py는 현재 불필요: generate_cxr.py가 모델을 한 번 로드하고 순차 생성한다.

## LINA 원본과의 관계

LINA 폴더는 별도로 유지한다. 지금 생성한 코드는 새 baseline 실행기이며 LINA 구현을 복사하지 않았다.
후속 개입 구현에서 참고/재사용한 파일·commit·license·변경 사항을 기록한다.
평가/마스크/기존 SD3.5·FLUX 개입 코드는 참고하되 Sana API에 맞게 이식한다.
