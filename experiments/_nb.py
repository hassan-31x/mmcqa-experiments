"""Notebook cell helpers shared by the experiment notebook generators (same setup cells as scripts/make_notebooks.py)."""

import json
from pathlib import Path

SETUP = '''
import glob, os, subprocess, sys
CODE = "/kaggle/working/mmcqa-experiments"
if not os.path.isdir(CODE):
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, REPO_URL, CODE], check=True)
print(subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
!nvidia-smi -L
!pip install -q -r {CODE}/requirements/qwen.txt
!pip list 2>/dev/null | grep -i -E "^(torch|transformers|peft|bitsandbytes|accelerate|huggingface-hub|tokenizers) "
'''

HF_TOKEN = '''
try:                                   # optional: a Kaggle secret HF_TOKEN avoids anonymous download limits
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    print("no HF_TOKEN secret: continuing with anonymous downloads")
'''

PACKAGE = '''
import shutil, pathlib
media = pathlib.Path(os.environ.get("MMCQA_MEDIA", "/kaggle/temp/mmcqa_media"))
for tmp in (WORK + "/mmcultureqa", "/kaggle/working/mmcqa_media", "/kaggle/working/mmcqa_envs", str(media)):
    shutil.rmtree(tmp, ignore_errors=True)
!cd /kaggle/working && zip -qr {ZIPNAME} results && ls -lh {ZIPNAME}
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def write_kernel(out: Path, cells, name: str, kernel_id: str, dataset_sources=()):
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {"id": kernel_id, "title": kernel_id.split("/")[1], "code_file": name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": list(dataset_sources),
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out / name)
