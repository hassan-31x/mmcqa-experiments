"""Phase 1 orchestrator: QLoRA-fine-tune Qwen3.5-4B (S1 and S2, one per GPU), then answer dev500 with each adapter.

    python -m mmcqa_exp.phase1 --workdir /kaggle/working/results --budget-hours 11 --train-hours 6
    python -m mmcqa_exp.phase1 --workdir /kaggle/working/results --limit 4          # smoke run
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import data, jobs as J, launcher, merge, settings as S
from .sft.configs import CONFIGS
from .util import dump_json, environment, hours_since, utc_now


def start_training(cfg_name, gpu, work, media, deadline):
    (Path(work) / "logs").mkdir(parents=True, exist_ok=True)
    log = (Path(work) / "logs" / f"sft_{cfg_name}_stdout.log").open("a")
    return subprocess.Popen(
        [sys.executable, "-m", "mmcqa_exp.sft.train", "--config", cfg_name, "--workdir", str(work),
         "--media", str(media), "--deadline", str(deadline)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(S.REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
             "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(S.REPO_ROOT)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/kaggle/working/results"))
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--budget-hours", type=float, default=S.DEFAULT_BUDGET_HOURS)
    ap.add_argument("--train-hours", type=float, default=6.0)
    ap.add_argument("--configs", nargs="*", default=["S1", "S2"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: items per track; uses the SMOKE config")
    ap.add_argument("--devtest", action="store_true", help="also answer devtest with the better adapter if time allows")
    args = ap.parse_args(argv)

    t0 = time.time()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = args.media or S.kaggle_media_dir()
    hard_deadline = t0 + args.budget_hours * 3600 - S.FINAL_RESERVE_MINUTES * 60
    gpus = launcher.list_gpu_indices()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    configs = ["SMOKE", "SMOKE2"][:len(gpus)] if args.limit else args.configs[:len(gpus)]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"Unknown config {c}; choose from {sorted(CONFIGS)}")
    print(f"[phase1] GPUs {gpus}; configs {configs}; media {media}", flush=True)

    # ---------------------------------------------------------------- prepare
    eval_set = data.eval_rows()
    if args.limit:
        eval_set = {t: rows[:args.limit] for t, rows in eval_set.items()}
    needed_dev = data.images_needed(eval_set) | data.images_needed(data.eval_loss_rows())
    data.fetch_images("dev", media, needed_dev)
    train_limit = CONFIGS[configs[0]].train_rows_per_track
    train_set = data.train_rows(train_limit)
    data.fetch_images("train", media, data.images_needed(train_set))
    data.official_repo(work)
    dump_json(work / "logs" / "phase1_run_config.json", {
        "started_utc": utc_now(), "args": {k: str(v) for k, v in vars(args).items()}, "configs": configs,
        "dataset_revision": S.DATASET_REVISION, "qwen": [S.QWEN_REPO, S.QWEN_REVISION],
        "qwen_max_pixels": S.QWEN_MAX_PIXELS, "qwen_max_new_tokens": S.QWEN_MAX_NEW_TOKENS,
        "eval_rows_per_track": len(eval_set["en"]), "environment": environment(),
        "sft_configs": {c: CONFIGS[c].as_dict() for c in configs}})

    # ---------------------------------------------------------------- train (one process per GPU)
    train_deadline = min(hard_deadline - 75 * 60, t0 + args.train_hours * 3600)
    if args.limit:
        train_deadline = t0 + 40 * 60
    procs = {}
    for stale in (work / "sft").glob("ready_*"):          # a leftover marker would defeat the staggered start
        stale.unlink()
    for k, cfg_name in enumerate(configs):
        procs[cfg_name] = start_training(cfg_name, gpus[k], work, media, train_deadline)
        ready = work / "sft" / f"ready_{cfg_name}"
        t_wait = time.time()
        while not ready.exists() and procs[cfg_name].poll() is None and time.time() - t_wait < 40 * 60:
            time.sleep(5)          # staggered start: the next trainer loads only after this one is ready
        print(f"[phase1] trainer {cfg_name} on GPU {gpus[k]}: "
              f"{ready.read_text() if ready.exists() else 'not ready'}", flush=True)
    for cfg_name, proc in procs.items():
        proc.wait()
        print(f"[phase1] trainer {cfg_name} exited with code {proc.returncode}", flush=True)

    summaries = {}
    for cfg_name in configs:
        path = work / "logs" / f"sft_{cfg_name}_summary.json"
        if path.exists():
            summaries[cfg_name] = json.loads(path.read_text())
    ok_configs = [c for c in configs if c in summaries and (work / "adapters" / c / "adapter_config.json").exists()]
    if not ok_configs:
        raise SystemExit("No adapter was trained. See logs/sft_*_failed.json and logs/sft_*_stdout.log.")

    # ---------------------------------------------------------------- answer dev500, one adapter per GPU
    specs = {c: J.StageSpec(name=f"sft_{c}", backend="qwen", prompt="p_new",
                            max_new_tokens=S.QWEN_MAX_NEW_TOKENS, adapter=f"adapters/{c}")
             for c in ok_configs}
    for spec in specs.values():
        J.build_plan(spec, work, eval_set)
    results, threads = {}, []

    def answer(cfg_name, spec, gpu):
        try:
            results[cfg_name] = launcher.run_stage(work, spec.name, sys.executable, [gpu], S.WORKERS_PER_GPU["qwen"],
                                                   hard_deadline, media)
        except Exception as exc:                 # keep the other adapter's run going
            results[cfg_name] = {"error": f"{type(exc).__name__}: {exc}"}

    for k, (cfg_name, spec) in enumerate(specs.items()):
        th = threading.Thread(target=answer, args=(cfg_name, spec, gpus[k % len(gpus)]))
        th.start()
        threads.append(th)
        time.sleep(1)
    for th in threads:
        th.join()

    status, checks = {}, {}
    for spec in specs.values():
        st = merge.merge_stage(work, spec.name)
        status.update(st)
        checks.update({f"{spec.name}|{k.split('|')[1]}": v for k, v in merge.check_stage_format(work, spec.name, st).items()})

    # ---------------------------------------------------------------- optional devtest
    devtest = None
    remaining_h = (hard_deadline - time.time()) / 3600
    if args.devtest and not args.limit and remaining_h > 3.0:
        best = min(ok_configs, key=lambda c: summaries[c].get("best_eval_loss") or 1e9)
        spec = J.StageSpec(name=f"devtest_sft_{best}", backend="qwen", prompt="p_new", split="devtest",
                           max_new_tokens=S.QWEN_MAX_NEW_TOKENS, adapter=f"adapters/{best}",
                           note="adapter chosen by eval loss, not by F1")
        rows = {t: data.load_rows("devtest", t) for t in S.TRACKS}
        data.fetch_images("devtest", media, data.images_needed(rows))
        J.build_plan(spec, work, rows)
        devtest = launcher.run_stage(work, spec.name, sys.executable, gpus, S.WORKERS_PER_GPU["qwen"], hard_deadline, media)
        st = merge.merge_stage(work, spec.name)
        status.update(st)
        checks.update({f"{spec.name}|{k.split('|')[1]}": v for k, v in merge.check_stage_format(work, spec.name, st).items()})

    merge.write_completion(work, "phase1_completion", {
        "finished_utc": utc_now(), "hours": round(hours_since(t0), 2), "training": summaries,
        "answering": results, "devtest": devtest, "status": status, "format_check": checks})
    print(f"\n[phase1] finished in {hours_since(t0):.2f} h")
    for key, st in status.items():
        print(f"  {key:30} {st['answered']:5}/{st['planned']:<5} errors={st['errors']} fallbacks={st['fallbacks']} "
              f"{checks.get(key, '').split('|')[0]}")


if __name__ == "__main__":
    main()
