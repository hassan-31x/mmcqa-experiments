"""Every pin and default in one place. Nothing here imports torch or transformers."""

import os
from pathlib import Path

# ----------------------------------------------------------------------------- data
DATASET_REPO = "QCRI/MMCQA-SemEval27"
DATASET_REVISION = "3090a38f8eb809a30dcf30a44ed653f77741c056"   # same pin as mmcqa-baseline
REGION, TASK = "mena", "qa"
TRACKS = ("en", "msa", "arz", "ajp")
VARIETY = {"en": "English", "msa": "Modern Standard Arabic",
           "arz": "Egyptian Arabic", "ajp": "Levantine Arabic"}

OFFICIAL_REPO_URL = "https://github.com/mmcultureqa-semeval27/mmcultureqa.git"
OFFICIAL_COMMIT = "7ec66a47acb82195a288525560c4acb26cb8b7ba"

# ----------------------------------------------------------------------------- models
QWEN_REPO = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"      # from the manifests of our Qwen3.5-4B run
QWEN9B_REPO = "Qwen/Qwen3.5-9B"                                  # same family, loaded in NF4 like the 4B (about 6 GB)
QWEN9B_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"     # HF main when the experiment was added (2026-11-09)
PANGEA_REPO = "neulab/CulturalPangea-7B"
PANGEA_REVISION = "5df045f3d042a66ee07b340ea1ce8b50e96fa58f"    # revision recorded by the old Pangea run
LLAVA_NEXT_GIT = "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"

# ----------------------------------------------------------------------------- evaluation sets
EVAL_SPLIT = "dev"
EVAL_ROWS = 500                  # "dev500": the first 500 rows per track (contains dev100)
HISTORY_ROWS = 100               # "dev100": the items every earlier run answered
EVAL_LOSS_SLICE = (900, 950)     # dev rows used only for SFT eval loss; disjoint from dev500
SEED = 2027

# ----------------------------------------------------------------------------- generation
QWEN_MAX_PIXELS = 448 * 448      # used for z1, SFT training AND SFT inference: must stay identical
QWEN_MIN_PIXELS = 256 * 256
QWEN_MAX_NEW_TOKENS = 96
PANGEA_MAX_NEW_TOKENS = 128      # identical to the old Pangea run
FUSION_HINT_WORDS = 80           # cap on a candidate/hint quoted inside a prompt

# Worker processes per GPU. Pangea may be raised to 2 after the smoke run (see plans/02).
WORKERS_PER_GPU = {"qwen": 2, "pangea": 1, "qwen9b": 1}
# Rough seconds per item per worker, used only to decide whether a stage fits the budget
# before it has been measured. Updated from measurements as stages finish.
DEFAULT_SEC_PER_ITEM = {"qwen": 7.0, "pangea": 6.0, "qwen9b": 14.0}
SCALING_EFFICIENCY = {"qwen": 0.7, "pangea": 0.85, "qwen9b": 1.0}   # 2 workers on one T4 do not double throughput

DEFAULT_BUDGET_HOURS = 11.0
FINAL_RESERVE_MINUTES = 25       # merge + zip at the end of a session


def kaggle_media_dir() -> Path:
    """Images live outside /kaggle/working so they are not counted in the 20 GB output limit."""
    override = os.environ.get("MMCQA_MEDIA")
    if override:
        return Path(override)
    for candidate in ("/kaggle/temp", "/kaggle/tmp"):
        if Path(candidate).is_dir():
            return Path(candidate) / "mmcqa_media"
    if Path("/kaggle").is_dir():
        return Path("/kaggle/working/mmcqa_media")     # zipped output must exclude it (notebooks delete it)
    return Path.home() / ".cache" / "mmcqa_media"


REPO_ROOT = Path(__file__).resolve().parents[1]
