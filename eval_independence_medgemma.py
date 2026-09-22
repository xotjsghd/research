import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
MAX_NEW_TOKENS = 320

QUESTION = """Analyze this frontal chest radiograph using only the image.

Assess pleural effusion and cardiomegaly separately.
Do not infer the presence of one finding from the other finding.
For pleural effusion, consider either side.
Use indeterminate if the visible image evidence is insufficient.

State:
1. Pleural effusion: present, absent, or indeterminate.
2. Cardiomegaly: present, absent, or indeterminate.

Then briefly explain:
3. Pleural effusion evidence: describe the visible evidence supporting
   your pleural effusion assessment.
4. Cardiomegaly evidence: describe the visible evidence supporting
   your cardiomegaly assessment.
5. Limitations: mention relevant image or projection limitations, if any.

Make sure each assessment agrees with its explanation.
Do not assume clinical information that is not visible in the image.
Use the numbered format above. Do not output JSON.
"""


def image_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("outputs/06_finding_independence"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "evaluations/finding_independence_medgemma_v2"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"이미지 폴더가 없습니다: {args.images}")

    images = sorted(
        args.images.glob("p*_seed*.png"),
        key=natural_key,
    )
    if not images:
        raise SystemExit("평가할 p*_seed*.png 이미지가 없습니다.")

    # 개수 제한 없이 폴더 안의 모든 해당 이미지를 평가합니다.
    hashes = {p.name: image_hash(p) for p in images}
    dataset_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()

    run_settings = {
        "model": MODEL_ID,
        "model_commit": MODEL_REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "question": QUESTION,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dataset_sha256": dataset_hash,
    }

    output = args.out_dir / "responses.jsonl"
    completed = set()

    if output.exists():
        if not args.resume:
            raise SystemExit(
                f"기존 결과가 있습니다: {output}\n"
                "이어서 실행하려면 --resume을 붙이세요."
            )

        data = output.read_bytes()
        if data and not data.endswith(b"\n"):
            raise SystemExit(
                "마지막 저장 행이 불완전합니다. "
                "파일을 백업하고 마지막 행을 확인하세요."
            )

        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = Path(row["image"]).name

            if any(row.get(k) != v for k, v in run_settings.items()):
                raise SystemExit(
                    "기존 결과와 평가 설정 또는 데이터가 다릅니다. "
                    "새 --out-dir을 지정하세요."
                )
            if (
                name not in hashes
                or row.get("image_sha256") != hashes[name]
                or not isinstance(row.get("response"), str)
            ):
                raise SystemExit(f"기존 결과 확인이 필요합니다: {name}")
            if name in completed:
                raise SystemExit(f"중복 결과가 있습니다: {name}")
            completed.add(name)

    print(
        f"전체 {len(images)}장 / 완료 {len(completed)}장 / "
        f"남음 {len(images) - len(completed)}장",
        flush=True,
    )

    if len(completed) == len(images):
        print(f"평가가 이미 완료되었습니다: {output}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU를 사용할 수 없습니다.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16을 지원하는 GPU가 필요합니다.")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with output.open("a", encoding="utf-8") as handle:
        for path in images:
            if path.name in completed:
                continue

            with Image.open(path) as source:
                image = source.convert("RGB")

            # 모델에는 이미지와 고정된 평가 질문만 전달합니다.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": QUESTION},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_length = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            tokens = generated[0, input_length:]
            response = processor.decode(
                tokens,
                skip_special_tokens=True,
            ).strip()

            hit_token_limit = (
                len(tokens) >= MAX_NEW_TOKENS
                and int(tokens[-1]) not in eos_ids
            )

            record = {
                **run_settings,
                "image": path.as_posix(),
                "image_sha256": hashes[path.name],
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "response": response,
                "generated_tokens": len(tokens),
                "hit_token_limit": hit_token_limit,
            }

            # 판정 파싱과 무관하게 원문을 먼저 보존합니다.
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed.add(path.name)

            print(
                f"\n[{len(completed)}/{len(images)}] {path.name}",
                flush=True,
            )
            print(response, flush=True)
            if hit_token_limit:
                print("[확인 필요] 응답이 토큰 한도에 도달했습니다.")

            del inputs, generated, tokens

    print(f"\n평가 완료: {output}")


if __name__ == "__main__":
    main()