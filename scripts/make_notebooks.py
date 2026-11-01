"""Generate the two Kaggle notebooks (one per phase). Re-run after editing, then push with `kaggle kernels push -p kaggle/phaseN`.

    python scripts/make_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


SETUP = '''
import glob, os, subprocess, sys
CODE = "/kaggle/working/mmcqa-experiments"
if not os.path.isdir(CODE):
    local = glob.glob("/kaggle/input/**/mmcqa_exp/__init__.py", recursive=True)    # a Kaggle Dataset copy also works
    if local:
        subprocess.run(["cp", "-r", os.path.dirname(os.path.dirname(local[0])), CODE], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--branch", BRANCH, REPO_URL, CODE], check=True)
print(subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
!nvidia-smi -L
!pip install -q -r {CODE}/requirements/qwen.txt
if globals().get("INSTALL_FLA", False):        # optional fast kernels for Qwen3.5's linear-attention layers
    !pip install -q flash-linear-attention
    print(subprocess.run([sys.executable, "-c", "import fla; print('fla', fla.__version__)"], capture_output=True, text=True))
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
for tmp in (WORK + "/mmcultureqa", "/kaggle/working/mmcqa_media", "/kaggle/working/mmcqa_envs", str(media), str(media.parent / "mmcqa_envs")):
    shutil.rmtree(tmp, ignore_errors=True)
!cd /kaggle/working && zip -qr {ZIPNAME} results && ls -lh {ZIPNAME}
'''

p1 = notebook([
    md("""
# Phase 1: QLoRA fine-tuning of Qwen3.5-4B on the MMCultureQA train split (SemEval 2027 Task 6, Task 2)

Trains two adapters at once (S1: lr 1e-4 on GPU 0, S2: lr 2e-4 on GPU 1), then answers the first 500 dev items per track
with each adapter, using the exact prompt, pixel budget and token cap of the zero-shot control in Phase 2.
Everything is generated here; scoring is done locally. Download `phase1_results.zip` from the Output tab.

**Before running:** Settings -> Accelerator -> **GPU T4 x2**, Internet **on**. Do a smoke run first (`LIMIT = 4`, about 40 min),
then set `LIMIT = None` and **Save Version -> Save & Run All (Commit)**. A commit keeps running with the browser closed.
Run Phase 2 in a separate session afterwards (one account: they cannot run at the same time).
"""),
    md("## 1. Config"),
    code('''
REPO_URL = "https://github.com/hassan-31x/mmcqa-experiments.git"
BRANCH = "main"
LIMIT = 4              # smoke test: 4 dev items per track and a 10-step training. Set to None for the full run.
BUDGET_HOURS = 11.0    # the run stops starting new work after this (Kaggle's limit is 12 h)
TRAIN_HOURS = 6.0      # training time per adapter; the rest of the budget answers dev500
DEVTEST = False        # True: also answer devtest with the better adapter if >= 3 h remain
INSTALL_FLA = True     # install flash-linear-attention (Triton kernels); the smoke run shows if it is faster on T4
WORK = "/kaggle/working/results"
'''),
    md("## 2. Code and dependencies"),
    code(SETUP),
    code(HF_TOKEN),
    md("## 3. Train both adapters, then answer dev500\nProgress is printed by the orchestrator. Logs: `results/logs/`."),
    code('''
args = f"--workdir {WORK} --budget-hours {BUDGET_HOURS} --train-hours {TRAIN_HOURS}"
if LIMIT: args += f" --limit {LIMIT}"
if DEVTEST: args += " --devtest"
!cd {CODE} && python -m mmcqa_exp.phase1 {args}
'''),
    md("## 4. What happened"),
    code('''
import json, glob
for p in sorted(glob.glob(f"{WORK}/logs/sft_*_sizing.json")) + sorted(glob.glob(f"{WORK}/logs/sft_*_summary.json")):
    print(os.path.basename(p), json.dumps(json.load(open(p)), indent=1)[:900])
for p in sorted(glob.glob(f"{WORK}/logs/sft_*_failed.json")):
    print("FAILED:", p, json.load(open(p))["error"])
print(open(f"{WORK}/logs/phase1_completion.json").read()[:3000])
for p in sorted(glob.glob(f"{WORK}/predictions/sft_*/dev_arz.jsonl")):
    print(p); [print("  ", json.loads(l)["question"], "->", json.loads(l)["answer"]) for l in open(p).readlines()[:3]]
'''),
    md("## 5. Package\nAdapters, logs, predictions and submission-format files in one zip. Images and the starter-kit clone are excluded."),
    code('ZIPNAME = "phase1_results.zip"\n' + PACKAGE),
])

p2 = notebook([
    md("""
# Phase 2: CulturalPangea-7B + Qwen3.5-4B candidates for the ensembles (SemEval 2027 Task 6, Task 2)

Stages on the first 500 dev items per track: `pangea_p0` (reproduces the earlier Pangea run), `qwen_z1` (zero-shot control
for Phase 1), `cascade_qwen2pangea` (Pangea with Qwen's answer as a hint), `fusion_qwen` (Qwen fuses both answers), and
optionally two sampled Qwen candidates for MBR. Pangea runs in its own virtualenv (transformers 4.37), Qwen in the main
environment (transformers 5.x). Download `phase2_results.zip` from the Output tab.

**Before running:** Settings -> Accelerator -> **GPU T4 x2**, Internet **on**. Smoke run first (`LIMIT = 4`), then the full run.
"""),
    md("## 1. Config"),
    code('''
REPO_URL = "https://github.com/hassan-31x/mmcqa-experiments.git"
BRANCH = "main"
LIMIT = 4              # smoke test: 4 dev items per track. Set to None for the full run.
BUDGET_HOURS = 11.0
SAMPLES = False        # True: two extra sampled Qwen candidates (about +1.5 h) for MBR
DEVTEST = False        # True: also answer devtest with pangea_p0 and qwen_z1 if time allows
PANGEA_WORKERS = None  # set to 2 after the smoke run if Pangea's peak_mem_gb is below ~7 GB
WORK = "/kaggle/working/results"
'''),
    md("## 2. Code and dependencies (main = Qwen environment; the Pangea venv is built and verified by the orchestrator)"),
    code(SETUP),
    code(HF_TOKEN),
    md("## 3. Run the stages"),
    code('''
args = f"--workdir {WORK} --budget-hours {BUDGET_HOURS}"
if LIMIT: args += f" --limit {LIMIT}"
if SAMPLES: args += " --samples"
if DEVTEST: args += " --devtest"
if PANGEA_WORKERS: args += f" --pangea-workers {PANGEA_WORKERS}"
!cd {CODE} && python -m mmcqa_exp.phase2 {args}
'''),
    md("## 4. What happened"),
    code('''
import json, glob
print(open(f"{WORK}/logs/phase2_completion.json").read()[:6000])
print(open(f"{WORK}/logs/phase2_run_config.json").read()[:2500])
for p in sorted(glob.glob(f"{WORK}/predictions/*/dev_arz.jsonl")):
    print(p); [print("  ", json.loads(l)["question"], "->", json.loads(l)["answer"]) for l in open(p).readlines()[:2]]
'''),
    md("## 5. Package"),
    code('ZIPNAME = "phase2_results.zip"\n' + PACKAGE),
])

for phase, nb, name in (("phase1", p1, "kaggle_phase1_finetune.ipynb"), ("phase2", p2, "kaggle_phase2_ensemble.ipynb")):
    out = ROOT / "kaggle" / phase
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {"id": f"muhammadhassan31/mmcqa-{phase}-" + ("finetune" if phase == "phase1" else "ensemble"),
            "title": f"mmcqa-{phase}-" + ("finetune" if phase == "phase1" else "ensemble"),
            "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out / name)
