"""Phase 3: does a quantized Qwen3.5-9B beat the 4B zero-shot? One stage on the first N dev items per track.

    python -m mmcqa_exp.phase3 --workdir /kaggle/working/results --rows 100
    python -m mmcqa_exp.phase3 --workdir /kaggle/working/results --rows 4        # smoke run

The stage `qwen9b_z1` uses the same prompt (p_new), pixel budget and token cap as `qwen_z1` (Phase 2), so the
only difference is the model: Qwen3.5-9B in NF4 (about 6 GB) instead of Qwen3.5-4B in NF4 (about 3 GB).
"""

import argparse
import sys
import time
from pathlib import Path

from . import data, jobs as J, launcher, merge, settings as S
from .util import dump_json, environment, hours_since, utc_now


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/kaggle/working/results"))
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=S.HISTORY_ROWS, help="items per track (the first N dev rows)")
    ap.add_argument("--budget-hours", type=float, default=5.0)
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--backend", default="qwen9b", help=argparse.SUPPRESS)      # "fake" in the tests
    args = ap.parse_args(argv)

    t0 = time.time()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = args.media or S.kaggle_media_dir()
    deadline = t0 + args.budget_hours * 3600 - S.FINAL_RESERVE_MINUTES * 60
    gpus = launcher.list_gpu_indices()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    wpg = args.workers_per_gpu or S.WORKERS_PER_GPU.get(args.backend, 1)

    eval_set = data.eval_rows(n=args.rows)
    data.fetch_images("dev", media, data.images_needed(eval_set))
    data.official_repo(work)
    spec = J.StageSpec("qwen9b_z1", args.backend, "p_new", S.QWEN_MAX_NEW_TOKENS,
                       note="Qwen3.5-9B NF4, same prompt/pixels/cap as qwen_z1 (the 4B zero-shot control)")
    dump_json(work / "logs" / "phase3_run_config.json", {
        "started_utc": utc_now(), "args": {k: str(v) for k, v in vars(args).items()},
        "dataset_revision": S.DATASET_REVISION, "model": [S.QWEN9B_REPO, S.QWEN9B_REVISION],
        "qwen_max_pixels": S.QWEN_MAX_PIXELS, "max_new_tokens": S.QWEN_MAX_NEW_TOKENS, "workers_per_gpu": wpg,
        "items_per_track": args.rows, "environment": environment(), "stage": spec.describe()})

    J.build_plan(spec, work, eval_set)
    n = sum(len(r) for r in eval_set.values())
    print(f"[phase3] {spec.name}: {n} items on GPUs {gpus}, {wpg} worker(s) per GPU", flush=True)
    res = launcher.run_stage(work, spec.name, sys.executable, gpus, wpg, deadline, media)
    status = merge.merge_stage(work, spec.name)
    checks = merge.check_stage_format(work, spec.name, status)
    merge.write_completion(work, "phase3_completion", {
        "hours": round(hours_since(t0), 2), "run": res, "status": status, "format_check": checks,
        "gpus": len(gpus), "workers_per_gpu": wpg})
    print(f"\n[phase3] finished in {hours_since(t0):.2f} h")
    for key, st in status.items():
        print(f"  {key:28} {st['answered']}/{st['planned']} errors={st['errors']} fallbacks={st['fallbacks']} "
              f"latency={st['mean_latency']}s peak_mem={st['peak_mem_gb']} GB")


if __name__ == "__main__":
    main()
