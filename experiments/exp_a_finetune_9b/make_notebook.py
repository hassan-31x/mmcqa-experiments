"""Generate the Kaggle notebook for experiment A:  python experiments/exp_a_finetune_9b/make_notebook.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _nb import HF_TOKEN, PACKAGE, SETUP, code, md, write_kernel   # noqa: E402

cells = [
    md("""
# Experiment A: fine-tune the quantized Qwen3.5-9B (NF4) with the 4B's recipe (SemEval 2027 Task 6, Task 2)

S1_9B (lr 1e-4, GPU 0) and S2_9B (lr 2e-4, GPU 1), each capped at the 4B's 870 steps (13,920 examples), time-budgeted.
Afterwards each adapter answers the first 100 dev items per track. Scoring and the judge run locally.
Download `exp_a_results.zip` from the Output tab. **Settings -> Accelerator -> GPU T4 x2, Internet on.**
"""),
    code('''
REPO_URL = "https://github.com/hassan-31x/mmcqa-experiments.git"
BRANCH = "main"
BUDGET_HOURS = 11.0
TRAIN_HOURS = 9.5
ROWS = 100
WORK = "/kaggle/working/results"
'''),
    code(SETUP),
    code(HF_TOKEN),
    code('''
!cd {CODE} && python experiments/exp_a_finetune_9b/run_kaggle.py --workdir {WORK} --budget-hours {BUDGET_HOURS} --train-hours {TRAIN_HOURS} --rows {ROWS}
'''),
    code('''
import json, glob
for p in sorted(glob.glob(f"{WORK}/logs/sft_*_sizing.json")) + sorted(glob.glob(f"{WORK}/logs/sft_*_summary.json")):
    print(os.path.basename(p), json.dumps(json.load(open(p)), indent=1)[:900])
for p in sorted(glob.glob(f"{WORK}/logs/sft_*_failed.json")):
    print("FAILED:", p, json.load(open(p))["error"])
print(open(f"{WORK}/logs/exp_a_completion.json").read()[:3000])
'''),
    code('ZIPNAME = "exp_a_results.zip"\n' + PACKAGE),
]
write_kernel(Path(__file__).parent / "kaggle", cells, "kaggle_exp_a_finetune_9b.ipynb",
             "muhammadhassan31/mmcqa-exp-a-finetune-9b")
