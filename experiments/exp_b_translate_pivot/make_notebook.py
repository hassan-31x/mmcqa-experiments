"""Generate the Kaggle notebook for experiment B:  python experiments/exp_b_translate_pivot/make_notebook.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _nb import HF_TOKEN, PACKAGE, SETUP, code, md, write_kernel   # noqa: E402

DATASET = "muhammadhassan31/mmcqa-exp-b-inputs"
cells = [
    md("""
# Experiment B: translate-pivot for MSA / Egyptian / Levantine (SemEval 2027 Task 6, Task 2)

The fine-tuned Qwen3.5-4B (S2 adapter) answers the English-translated questions in English (translation of questions and of
answers happens locally via OpenRouter; this notebook is only the GPU step). Input dataset: `mmcqa-exp-b-inputs`.
Download `exp_b_results.zip` from the Output tab. **Settings -> Accelerator -> GPU T4 x2, Internet on.**
"""),
    code('''
REPO_URL = "https://github.com/hassan-31x/mmcqa-experiments.git"
BRANCH = "main"
ROWS = 100
WORK = "/kaggle/working/results"
'''),
    code(SETUP),
    code(HF_TOKEN),
    code('''
!cd {CODE} && python experiments/exp_b_translate_pivot/run_kaggle.py --workdir {WORK} --inputs /kaggle/input --rows {ROWS}
'''),
    code('''
import json, glob
print(open(f"{WORK}/logs/exp_b_completion.json").read()[:2500])
for p in sorted(glob.glob(f"{WORK}/predictions/pivot_en_s2/dev_*.jsonl")):
    print(p); [print("  ", json.loads(l)["question"], "->", json.loads(l)["answer"]) for l in open(p).readlines()[:3]]
'''),
    code('ZIPNAME = "exp_b_results.zip"\n' + PACKAGE),
]
write_kernel(Path(__file__).parent / "kaggle", cells, "kaggle_exp_b_translate_pivot.ipynb",
             "muhammadhassan31/mmcqa-exp-b-translate-pivot", dataset_sources=[DATASET])
