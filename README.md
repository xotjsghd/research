# medical-lina: 첫 CXR 생성

현재 포함: 환경 확인, CXR용 Sana 단일/반복 생성, 실행 기록, 향후 구조.
아직 미포함: 의료 평가, c_biased, LINA 개입, 감마 탐색, AIM.
이 폴더는 연구실 서버에 자동 설치되지 않는다. 기존 LINA와 나란히 별도 폴더로 옮긴다.

## 1. 파일 배치와 환경 확인

압축을 노트북에서 풀고 원격 VS Code 탐색기로 medical-lina 폴더를
`/LocalData1/tshong/` 아래 업로드한 뒤 그 폴더를 연다.
원격 터미널에서:

```bash
cd /LocalData1/tshong/medical-lina
python check_environment.py
```

python이 없다면 python3로 실행한다. 먼저 이 출력을 확인하고 전용 환경의 설치를 결정한다.
공유 Python 환경에 설치하지 않는다. 설치 스크립트가 자동 실행되지는 않는다.
서버의 Blackwell GPU를 지원하는 PyTorch를 먼저 선택해야 한다.
`nvidia-smi`의 CUDA 버전은 드라이버 정보이며 설치된 torch의 CUDA 버전과 같은 뜻이 아니다.
PyTorch 설치: https://pytorch.org/get-started/locally/
호환되는 PyTorch가 설치된 개인 환경에서 나머지는 다음으로 설치한다:

```bash
python -m pip install -r requirements.txt
```

requirements는 시작용 범위이며 GPU 검증된 고정 환경은 아니다.

## 2. 다운로드 없이 입력 확인

```bash
python generate_cxr.py --prompts prompts/smoke_test.csv --seeds 0 1 2 3 --dry-run
```

12개 생성 작업의 설정을 출력한다. 모델 다운로드·GPU 사용·결과 파일 생성은 없다.
샘플 문장들은 수작업 동작 확인용이다. 단측 소견의 언급이 반대쪽 소견 부재를
명시하는 것은 아니므로, 임상 평가용 기대 조건은 추후 별도 정의한다.

## 3. 한 장 생성

gpujj로 확인한 빈 GPU 번호를 직접 지정한다. 아래 2는 예시일 뿐이다.

```bash
CUDA_VISIBLE_DEVICES=2 python generate_cxr.py --prompt "A frontal chest radiograph showing a left pleural effusion." --seeds 42 --cache-dir /LocalData1/tshong/model-cache
```

첫 실행은 모델을 다운로드하며 용량·접근 권한·네트워크 상태에 따라 실패할 수 있다.
CPU fallback은 하지 않는다. 프로그램 내부 cuda:0은 여기서 지정한 서버 GPU 2다.
기본값: 512px, 20 steps, CFG 4.5. 논문 결과 재현용 최적값이 아니라 동작 확인용이다.
스케줄러와 prompt instruction은 로드한 pipeline의 기본 동작을 유지한다.
VAE/text encoder는 bf16(미지원시 fp32), transformer는 fp16으로 시작한다.
체크포인트가 fp32로 공개되어 있으므로 로드 시 CPU RAM과 다운로드 공간이 필요하다.

## 4. 반복 생성

```bash
CUDA_VISIBLE_DEVICES=2 python generate_cxr.py --prompts prompts/smoke_test.csv --seeds 0 1 2 3 --cache-dir /LocalData1/tshong/model-cache
```

모델은 한 번만 로드하고 한 장씩 순차 생성한다. 실행마다 outputs 안에 고유 폴더를 만든다.
PNG, 프롬프트/seed/이미지 연결 manifest.jsonl, 설정·패키지 버전·scheduler·상태 run.json을 저장한다.
중간 실패 시 이미 저장한 이미지와 기록을 남긴다. 자동 재개는 아직 없다.
같은 seed도 소프트웨어·GPU 변경 시 완전히 같은 픽셀을 보장하지 않는다.
비교 실험부터는 --revision에 모델 commit hash를 고정하고 패키지 버전도 고정한다.

## 범위와 검증

현재 파일은 GPU 없는 환경에서 문법·dry-run·입력 검증을 점검한다.
실제 모델 다운로드와 GPU 이미지 생성은 연구실 서버에서 확인해야 한다.
전체 파일 계획: PROJECT_PLAN.md.

## 참고

- CXR checkpoint: https://huggingface.co/raman07/CheXGenBench-Models-Sana-e20
- Sana API / precision: https://huggingface.co/docs/diffusers/api/pipelines/sana
- CheXGenBench: https://github.com/Raman1121/CheXGenBench
- LINA: https://github.com/OpenCausaLab/LINA
