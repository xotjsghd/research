"""Read-only environment report. No installation, download or CUDA allocation."""
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

report = {'python': sys.version, 'executable': sys.executable,
          'platform': platform.platform(), 'cwd': os.getcwd(),
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'packages': {}}
for package in ['torch', 'diffusers', 'transformers', 'accelerate', 'huggingface-hub', 'Pillow', 'sentencepiece']:
    try:
        report['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][package] = 'NOT INSTALLED'
if shutil.which('nvidia-smi'):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                 '--format=csv'], capture_output=True, text=True, timeout=15)
        report['gpus'] = result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        report['gpus'] = 'nvidia-smi timed out'
print(json.dumps(report, ensure_ascii=False, indent=2))
