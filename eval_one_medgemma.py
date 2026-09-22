from pathlib import Path

import torch
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

MODEL_ID = "google/medgemma-1.5-4b-it"
IMAGE_PATH = Path(
    "outputs/04_effusion_order_swap/p000_seed0.png"
)
CACHE_DIR = "/LocalData1/tshong/model-cache"

question = """Analyze this frontal chest radiograph using only the image.

State:
1. Patient right pleural effusion: absent, small, moderate, large, or indeterminate.
2. Patient left pleural effusion: absent, small, moderate, large, or indeterminate.
3. Relative size: right larger, left larger, similar, or indeterminate.

Briefly explain the image evidence. Do not assume any clinical information that is not visible in the image."""

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
).eval()

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }
]

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

answer = processor.decode(
    output_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print("\n--- MedGemma evaluation ---\n")
print(answer)