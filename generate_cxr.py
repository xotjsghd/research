"""Baseline CXR generation. No LINA intervention or medical evaluation yet."""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='One English prompt')
    source.add_argument('--prompts', type=Path, help='CSV with id,prompt columns')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--guidance', type=float, default=4.5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--model', default='raman07/CheXGenBench-Models-Sana-e20')
    p.add_argument('--revision', help='Optional Hugging Face commit hash')
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--cache-dir', type=Path, help='Model download cache directory')
    p.add_argument('--transformer-dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--dry-run', action='store_true', help='Validate and print jobs; no downloads or GPU use')
    a = p.parse_args()
    if a.steps < 1 or a.size < 32 or a.size % 32:
        p.error('steps must be positive; size must be a positive multiple of 32')
    if not math.isfinite(a.guidance) or a.guidance < 0:
        p.error('guidance must be finite and nonnegative')
    if any(s < 0 or s >= 2**63 for s in a.seeds):
        p.error('seeds must be in [0, 2**63)')
    if len(set(a.seeds)) != len(a.seeds):
        p.error('seeds must be unique')
    return a


def read_prompts(a):
    if a.prompt is not None:
        rows = [{'id': 'single', 'prompt': a.prompt}]
    else:
        with a.prompts.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'id', 'prompt'} <= set(reader.fieldnames or []):
                raise ValueError('CSV requires id,prompt columns')
            rows = list(reader)
    if not rows or any(not (r.get('id') or '').strip() or not (r.get('prompt') or '').strip() for r in rows):
        raise ValueError('At least one nonempty id and prompt is required')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Prompt ids must be unique')
    return [{'id': r['id'], 'prompt': r['prompt']} for r in rows]


def write_json(path, data):
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def main():
    a = parse_args()
    rows = read_prompts(a)
    print(f'{len(rows)} prompts x {len(a.seeds)} seeds = {len(rows) * len(a.seeds)} images')
    if a.dry_run:
        print(json.dumps({'settings': vars(a), 'prompts': rows}, ensure_ascii=False, indent=2, default=str))
        return
    # GPU choice must be explicit; never automatically claim an idle shared GPU.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        raise SystemExit('Select one GPU first, e.g. CUDA_VISIBLE_DEVICES=2 python generate_cxr.py ...')
    import torch
    from diffusers import SanaPipeline
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check the selected environment and GPU-compatible PyTorch build.')
    if torch.cuda.device_count() != 1:
        raise SystemExit('Exactly one visible GPU is required.')
    # A real kernel check catches unsupported GPU/PyTorch combinations before download.
    (torch.ones((2, 2), device='cuda') @ torch.ones((2, 2), device='cuda')).sum().item()
    if a.transformer_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('Selected GPU does not support bfloat16.')
    run = a.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ['torch', 'diffusers', 'transformers', 'accelerate', 'huggingface-hub', 'Pillow']:
        versions[package] = importlib.metadata.version(package)
    meta = {'settings': vars(a), 'prompts': rows, 'versions': versions,
            'cuda_visible_devices': visible, 'gpu': torch.cuda.get_device_name(0),
            'torch_cuda': torch.version.cuda, 'status': 'loading',
            'note': 'Smoke-test settings, not a verified reproduction of CheXGenBench.'}
    write_json(run / 'run.json', meta)
    print(f'Output: {run.resolve()}', flush=True)
    try:
        kwargs = {'torch_dtype': torch.float32, 'use_safetensors': True}
        if a.revision:
            kwargs['revision'] = a.revision
        if a.cache_dir:
            kwargs['cache_dir'] = str(a.cache_dir)
        pipe = SanaPipeline.from_pretrained(a.model, **kwargs)
        # Sana docs: keep text encoder and VAE in bf16/fp32, not fp16.
        pipe.transformer.to(dtype=getattr(torch, a.transformer_dtype))
        safe_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        pipe.text_encoder.to(dtype=safe_dtype)
        pipe.vae.to(dtype=safe_dtype)
        pipe.to('cuda')
        meta.update(status='generating', scheduler=type(pipe.scheduler).__name__,
                    scheduler_config=dict(pipe.scheduler.config), pipeline_config=dict(pipe.config),
                    text_encoder_dtype=str(safe_dtype), vae_dtype=str(safe_dtype))
        write_json(run / 'run.json', meta)
        with (run / 'manifest.jsonl').open('a', encoding='utf-8') as manifest:
            for index, row in enumerate(rows):
                for seed in a.seeds:
                    name = f'p{index:03d}_seed{seed}.png'
                    generator = torch.Generator(device='cpu').manual_seed(seed)
                    with torch.inference_mode():
                        result = pipe(prompt=row['prompt'], negative_prompt='',
                                      num_inference_steps=a.steps, guidance_scale=a.guidance,
                                      height=a.size, width=a.size, generator=generator,
                                      use_resolution_binning=False)
                    result.images[0].save(run / name)
                    record = dict(row, seed=seed, image=name, width=result.images[0].width,
                                  height=result.images[0].height)
                    manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                    manifest.flush()
                    print(f'Saved {name}', flush=True)
        meta['status'] = 'completed'
    except BaseException as exc:
        meta.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        write_json(run / 'run.json', meta)


if __name__ == '__main__':
    main()
