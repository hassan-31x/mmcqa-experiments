"""Experiment B, GPU step: the fine-tuned 4B (S2) answers the *English-translated* questions of msa/arz/ajp.

    python experiments/exp_b_translate_pivot/run_kaggle.py --workdir /kaggle/working/results --inputs /kaggle/input/<dataset>

Inputs (a Kaggle dataset built by build_dataset.sh): questions_en.jsonl and adapters/S2 (the Phase 1 adapter).
The prompt is the training prompt (p_new) in English, so the model answers in English whatever the original track.
The direct baseline ("raw answer from and to that language") is sft_S2 from Phase 1 on the very same items.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mmcqa_exp import data, jobs as J, launcher, merge, settings as S           # noqa: E402
from mmcqa_exp.util import dump_json, environment, hours_since, utc_now         # noqa: E402

ARABIC = ("msa", "arz", "ajp")


def find_inputs(root: Path) -> Path:
    hits = list(root.rglob("questions_en.jsonl"))
    if not hits:
        raise SystemExit(f"questions_en.jsonl not found under {root}")
    return hits[0].parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/kaggle/working/results"))
    ap.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=S.HISTORY_ROWS)
    ap.add_argument("--budget-hours", type=float, default=3.0)
    args = ap.parse_args(argv)

    t0 = time.time()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = args.media or S.kaggle_media_dir()
    deadline = t0 + args.budget_hours * 3600 - S.FINAL_RESERVE_MINUTES * 60
    gpus = launcher.list_gpu_indices()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    inp = find_inputs(args.inputs)
    for z in inp.glob("*.zip"):                       # a zipped folder that Kaggle did not extract
        shutil.unpack_archive(str(z), str(work / "_unz"))
    cfg = next(iter(list(inp.rglob("adapter_config.json")) + list((work / "_unz").rglob("adapter_config.json"))), None)
    if cfg is None:
        raise SystemExit(f"adapter_config.json not found under {inp}")
    shutil.copytree(cfg.parent, work / "adapters" / "S2", dirs_exist_ok=True)
    shutil.copy(inp / "questions_en.jsonl", work / "questions_en.jsonl")
    q_en = {(r["track"], r["id"]): r["question_en"]
            for r in map(json.loads, (work / "questions_en.jsonl").read_text(encoding="utf-8").splitlines())}

    rows = {t: [dict(r, question=q_en[(t, r["id"])]) for r in rows_t] if t in ARABIC else []
            for t, rows_t in data.eval_rows(n=args.rows).items()}
    missing = [(t, r["id"]) for t in ARABIC for r in rows[t] if (t, r["id"]) not in q_en]
    assert not missing, f"{len(missing)} questions have no English translation"
    data.fetch_images("dev", media, data.images_needed(rows))
    data.official_repo(work)
    spec = J.StageSpec("pivot_en_s2", "qwen", "p_new_en", S.QWEN_MAX_NEW_TOKENS, adapter="adapters/S2",
                       note="fine-tuned 4B (S2) answers the English-translated question in English")
    dump_json(work / "logs" / "exp_b_run_config.json", {
        "started_utc": utc_now(), "args": {k: str(v) for k, v in vars(args).items()}, "stage": spec.describe(),
        "items": {t: len(rows[t]) for t in S.TRACKS}, "dataset_revision": S.DATASET_REVISION,
        "model": [S.QWEN_REPO, S.QWEN_REVISION], "environment": environment()})
    J.build_plan(spec, work, rows)
    res = launcher.run_stage(work, spec.name, sys.executable, gpus, S.WORKERS_PER_GPU["qwen"], deadline, media)
    status = merge.merge_stage(work, spec.name)
    merge.write_completion(work, "exp_b_completion", {"hours": round(hours_since(t0), 2), "run": res, "status": status})
    print(f"\n[exp_b] finished in {hours_since(t0):.2f} h")
    for k, st in status.items():
        print(f"  {k:24} {st['answered']}/{st['planned']} errors={st['errors']} fallbacks={st['fallbacks']}")


if __name__ == "__main__":
    main()
