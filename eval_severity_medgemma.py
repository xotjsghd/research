import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MODEL_ID = "google/medgemma-1.5-4b-it"
CACHE_DIR = "/LocalData1/tshong/model-cache"

FOLDERS = [
    ROOT / "outputs/05_effusion_severity",
]   

SAVE_DIR = ROOT / "evaluations/effusion_severity_medgemma_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = SAVE_DIR / "responses.jsonl"

# 한 장 테스트와 동일한 질문. 생성 프롬프트는 모델에 제공하지 않음.
QUESTION = """Analyze this frontal chest radiograph using only the image.

State:
1. Patient right pleural effusion: absent, small, moderate, large, or indeterminate.
2. Patient left pleural effusion: absent, small, moderate, large, or indeterminate.
3. Relative size: right larger, left larger, similar, or indeterminate.

Briefly explain the image evidence. Do not assume any clinical information that is not visible in the image."""

# 가중치를 불러오기 전에 입력 파일부터 확인
images = []
for folder in FOLDERS:
    expected = [
        folder / f"p{condition:03d}_seed{seed}.png"
        for condition in range(6)
        for seed in range(50)
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"이미지가 없습니다: {missing[:5]}")
    images.extend(expected)

# 같은 출력 파일에서는 완료된 이미지를 건너뜀
done = set()
if RESULT_PATH.exists():
    with RESULT_PATH.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"responses.jsonl의 {line_number}번째 줄이 불완전합니다."
                )
            if record["question"] != QUESTION:
                raise RuntimeError("질문이 바뀌었습니다. SAVE_DIR을 새 이름으로 바꾸세요.")
            done.add(record["image"])

pending = [
    p for p in images
    if str(p.relative_to(ROOT)) not in done
]
print(f"전체 {len(images)}장 / 이번 실행 {len(pending)}장", flush=True)

if not pending:
    print("모두 평가 완료됐습니다.")
    raise SystemExit

processor = AutoProcessor.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR
)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
).eval()

metadata = {
    "model": MODEL_ID,
    "model_commit": getattr(model.config, "_commit_hash", None),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "question": QUESTION,
    "do_sample": False,
    "max_new_tokens": 180,
}

for index, path in enumerate(pending, 1):
    relative_path = str(path.relative_to(ROOT))
    print(f"[{index}/{len(pending)}] {relative_path}", flush=True)

    with Image.open(path) as source:
        image = source.convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QUESTION},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = processor.decode(generated, skip_special_tokens=True)

    record = {
        **metadata,
        "image": relative_path,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "response": answer,
        "generated_tokens": len(generated),
        "hit_token_limit": len(generated) >= 180,
    }

    # 이미지 한 장마다 저장
    with RESULT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    del inputs, output_ids, generated

print(f"완료: {RESULT_PATH}", flush=True)