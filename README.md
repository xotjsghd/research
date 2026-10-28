# Medical LINA — LINA를 흉부 X-ray 생성에 적용하기 위한 개인 탐색 연구

> 고려대학교 MILab 학부연구생 기간(2026.07–2026.10)에 진행한 **개인 탐색 연구**입니다.
> 문제 정의와 초기 실험 단계까지 진행했고, 방법(개입·AIM) 구현 전에 종료했습니다.
> 논문 출판이나 성능 개선 결과는 없습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 질문 | 텍스트→의료영상 생성모델은 **여러 소견 + 위치(좌/우) + 크기** 조건을 프롬프트대로 반영하는가? |
| 참고 방법 | LINA (Yu & Lu, arXiv 2512.13290) — 관계 토큰 개입과 적응형 개입 강도 예측 |
| 생성모델 | `raman07/CheXGenBench-Models-Sana-e20` (Sana, 512px) |
| 평가 | `google/medgemma-1.5-4b-it` 로컬 GPU 추론(bf16, greedy) |
| 진행 범위 | 설계안 작성 → 기준(baseline) 생성 → MedGemma 평가 → 조건 반영 오류 탐색 |
| 미구현 | 의료 규칙 구축, 토큰·샘플링 개입, γ 탐색, AIM 학습, 방법 비교 |

## 진행 과정

1. **논문 공부:** DDPM → DDIM → CFG → LDM → DiT → Flow Matching → SD3 순서로 읽고 박사과정 멘토와 일대일 미팅에서 발표했습니다.
2. **방법 선정:** 자연영상에서 프롬프트의 관계·물리 조건 반영을 개선한 LINA를 참고 방법으로 정했습니다.
3. **설계안 작성:** LINA의 2단계 구조를 흉부 X-ray에 맞게 재구성한 **MedPAP** 도식을 만들고 멘토 피드백을 받았습니다.
4. **기준 실험:** 개입 없이 Sana로 프롬프트당 50장(seed 0–49)을 생성하고, MedGemma로 판정해 어떤 조건이 잘 반영되지 않는지 찾았습니다.

## 참고 방법: LINA

![LINA Figure 6](docs/figures/lina_fig6.png)

*출처: Shu Yu, Chaochao Lu. "LINA: Learning INterventions Adaptively for Physical Alignment and Generalization in Diffusion Models." arXiv:2512.13290, Figure 6. 참고용으로 인용했으며 저작권은 원 저자에게 있습니다.*

- **Phase 1 (offline):** 실패 사례(hard case)를 모으고, MLLM 평가를 반복하며 사례별 최적 개입 강도 (γ₁\*, γ₂\*)를 찾아 AIM(T5 + Adapter)을 학습합니다.
- **Phase 2 (online):** 새 프롬프트에 대해 AIM이 γ를 예측하고, ① 관계 토큰 임베딩 개입(γ₁), ② neutral 프롬프트와의 대조 guidance(γ₂), ③ 초기 구조 형성 단계에 계산을 더 배분하는 방식으로 생성합니다.

## 설계안: MedPAP (구현 전 설계)

LINA Figure 6을 흉부 X-ray 도메인에 맞게 재구성한 도식입니다. 점선 주황색 상자가 의료 도메인에 맞게 추가·변경한 부분입니다.

### Phase 1 — 의료 감독 신호 구성과 AIM 학습

![MedPAP Phase 1](docs/figures/medpap_phase1.svg)

### Phase 2 — 적응형 개입을 적용한 생성

![MedPAP Phase 2](docs/figures/medpap_phase2.svg)

### LINA 원본과의 차이

| 구성 요소 | LINA (자연영상) | MedPAP 설계안 (흉부 X-ray) |
|---|---|---|
| 프롬프트·목표 | PAP 프롬프트, 파싱한 목표 CSG | Chest ImaGenome 기반 관계 후보 + 임상 검토를 거친 규칙 R, `[severity] [side] [finding]` 템플릿 |
| 생성모델 | SD3.5 / FLUX 등 | CheXGenBench-Sana-e20 (frozen) |
| 평가 | SAM 마스크 + MLLM | CXAS 해부학 마스크(위치·기하) + MAIRA-2 소견 생성·grounding |
| 프롬프트 파서 | spaCy로 관계 토큰 선택 | 소견·위치·중증도 조합을 고르는 의료 프롬프트 파서 |
| AIM | T5 + Adapter | Medical-mT5(frozen) + MLP, L2 회귀 |

> 실제 초기 실험에서는 멘토 피드백에 따라 평가기로 MedGemma를 먼저 사용했습니다. 도식의 CXAS·MAIRA-2 구성은 이후 검증 단계를 위한 계획입니다.

## 구현한 것

| 파일 | 역할 |
|---|---|
| `check_environment.py` | Python·패키지·GPU 상태 출력 (설치·다운로드 없음) |
| `generate_cxr.py` | 프롬프트(CSV) × seed → PNG, 생성 설정, manifest 저장. `--dry-run` 지원 |
| `eval_one_medgemma.py` | 이미지 1장 MedGemma 판정 (동작 확인용) |
| `eval_batch_medgemma.py` | 흉수 좌우·상대 크기 판정 (프롬프트 세트 03·04) |
| `eval_severity_medgemma.py` | 흉수 크기 3단계 × 좌우 판정 (세트 05) |
| `eval_independence_medgemma.py` | 흉수 × 심비대 독립성 판정 (세트 06), `--resume` 지원 |
| `prompts/*.csv` | 실험용 프롬프트 세트 01–07 |
| `evaluations/*/responses.jsonl` | 판정 결과 700건. 모델 commit, 라이브러리 버전, 질문, 응답 기록 |

설계 원칙: 평가 모델에는 생성 프롬프트를 주지 않고 이미지만 보고 판정하게 했습니다. 재현을 위해 모델 commit, 버전, seed, 이미지 경로를 결과마다 남겼습니다.

### 프롬프트 세트

| 세트 | 확인하려는 것 |
|---|---|
| 01 | 좌우 위치 기본 반영 |
| 02 | 흉수 표현 탐색 |
| 03 | 좌우 흉수 크기 조합 (작은 쪽 + 큰 쪽) |
| 04 | 03과 같은 의미, 어순만 교체 |
| 05 | 흉수 크기 3단계(small / moderate / large) × 좌우 |
| 06 | 흉수 × 심비대 2×2 (한 소견이 다른 소견을 끌고 오는가) |
| 07 | 흉수·기흉 좌우 결합 |

## 초기 관찰 (MedGemma 판정 기준)

아래 수치는 **평가 모델의 판정**이며 정답 라벨이 아닙니다.

- **크기 조건 (세트 05, 300장):** small 프롬프트에서 대상 쪽이 small로 판정된 경우는 0/100이었고, 대부분 moderate로 판정됐습니다. large 프롬프트의 large 판정 비율은 왼쪽 82%, 오른쪽 40%로 좌우 차이가 컸습니다.
- **어순 (세트 03·04, 200장):** 같은 의미라도 어순에 따라 상대 크기 판정이 달라졌습니다. 예: "작은 왼쪽 + 큰 오른쪽"에서 작은 소견을 먼저 쓰면 66%, 큰 소견을 먼저 쓰면 100%.
- **소견 독립성 (세트 06, 200장):** 흉수만 요청했을 때 심비대 판정 3/50, 심비대만 요청했을 때 흉수 판정 0/50. 흉수를 요청했는데 흉수 없음으로 판정된 경우 8/50.

## 한계와 배운 점

- **평가기 신뢰도:** small 판정이 거의 나오지 않는 것이 생성모델 문제인지, 평가모델의 판정 성향인지 구분할 수 없었습니다. 판정을 정답처럼 쓰려면 분할 기반 기하 측정 같은 독립적인 검증이 함께 필요합니다.
- **도메인 지식:** 생성 결과의 임상적 타당성을 스스로 판단하기 어려웠고, 임상 규칙(R) 구축에는 전문가 검토가 필요했습니다.
- 이 단계에서 연구를 종료했기 때문에 개입 방법의 효과는 검증하지 않았습니다.

## 실행 방법

```bash
# PyTorch는 GPU·드라이버에 맞는 버전을 별도 설치
pip install -r requirements.txt
python check_environment.py

# 기준 생성 (먼저 --dry-run으로 작업 목록 확인)
python generate_cxr.py --prompts prompts/05_effusion_severity.csv --seeds $(seq 0 49) --dry-run
python generate_cxr.py --prompts prompts/05_effusion_severity.csv --seeds $(seq 0 49)

# MedGemma 판정 (모델 캐시 경로 CACHE_DIR는 환경에 맞게 수정)
python eval_severity_medgemma.py
```

- 생성 이미지(`outputs/`)와 모델 가중치는 저장소에 포함하지 않았습니다.
- MedGemma는 Google Health AI Developer Foundations 이용약관에 동의한 후 사용할 수 있습니다.

## 저장소 구조

```
.
├── README.md
├── PROJECT_PLAN.md          # 전체 구현 계획 (단계별 파일·상태)
├── check_environment.py
├── generate_cxr.py
├── eval_*_medgemma.py
├── prompts/                 # 프롬프트 세트 01–07
├── evaluations/             # MedGemma 판정 결과 (jsonl)
└── docs/figures/            # LINA Figure 6, MedPAP 설계 도식
```

## 기여 범위

- 연구 질문 설정, MedPAP 설계, 실험 설계, 코드 작성(AI 코딩 도구 활용), 실험 실행과 결과 분석은 본인이 진행했습니다.
- 논문 학습 방향과 설계·평가 모델 선택에 대해 MILab 박사과정 멘토의 피드백을 받았습니다.

## 참고문헌

- Shu Yu, Chaochao Lu. *LINA: Learning INterventions Adaptively for Physical Alignment and Generalization in Diffusion Models.* arXiv:2512.13290. [Project page](https://opencausalab.github.io/LINA)
- CheXGenBench Sana checkpoint: [raman07/CheXGenBench-Models-Sana-e20](https://huggingface.co/raman07/CheXGenBench-Models-Sana-e20)
- MedGemma: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
