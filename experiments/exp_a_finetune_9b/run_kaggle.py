"""Experiment A: QLoRA fine-tune the quantized Qwen3.5-9B with the 4B's recipe, then answer dev100 with each adapter.

    python experiments/exp_a_finetune_9b/run_kaggle.py --workdir /kaggle/working/results --budget-hours 11 --train-hours 9.5

S1_9B (lr 1e-4) trains on GPU 0 and S2_9B (lr 2e-4) on GPU 1, mirroring the 4B's S1/S2. Both are capped at the 4B's 870
optimizer steps (batch 16 = 13,920 examples); if the time budget allows fewer, the sizing step shrinks them and the
fraction of the 4B's data actually seen is recorded in logs/sft_*_sizing.json. Then each adapter answers the first
``--rows`` dev items per track (default 100, the items every comparison uses) with the same prompt as the 4B runs.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mmcqa_exp import data, jobs as J, launcher, merge, phase1, settings as S          # noqa: E402
from mmcqa_exp.sft.configs import CONFIGS                                             # noqa: E402
from mmcqa_exp.util import dump_json, environment, hours_since, utc_now               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/kaggle/working/results"))
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--budget-hours", type=float, default=11.0)
    ap.add_argument("--train-hours", type=float, default=9.5)
    ap.add_argument("--configs", nargs="*", default=["S1_9B", "S2_9B"])
    ap.add_argument("--rows", type=int, default=S.HISTORY_ROWS)
    args = ap.parse_args(argv)

    t0 = time.time()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = args.media or S.kaggle_media_dir()
    hard_deadline = t0 + args.budget_hours * 3600 - S.FINAL_RESERVE_MINUTES * 60
    gpus = launcher.list_gpu_indices()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    configs = args.configs[:len(gpus)]
    print(f"[exp_a] GPUs {gpus}; configs {configs}; media {media}", flush=True)

    eval_set = data.eval_rows(n=args.rows)
    data.fetch_images("dev", media, data.images_needed(eval_set) | data.images_needed(data.eval_loss_rows()))
    train_set = data.train_rows(CONFIGS[configs[0]].train_rows_per_track)
    data.fetch_images("train", media, data.images_needed(train_set))
    data.official_repo(work)
    dump_json(work / "logs" / "exp_a_run_config.json", {
        "started_utc": utc_now(), "args": {k: str(v) for k, v in vars(args).items()}, "configs": configs,
        "model": [S.QWEN9B_REPO, S.QWEN9B_REVISION], "dataset_revision": S.DATASET_REVISION,
        "qwen_max_pixels": S.QWEN_MAX_PIXELS, "qwen_max_new_tokens": S.QWEN_MAX_NEW_TOKENS,
        "items_per_track": args.rows, "environment": environment(),
        "sft_configs": {c: CONFIGS[c].as_dict() for c in configs}})

    # ------------------------------------------------------------------ train, one process per GPU, staggered start
    train_deadline = min(hard_deadline - 60 * 60, t0 + args.train_hours * 3600)
    for stale in (work / "sft").glob("ready_*"):
        stale.unlink()
    procs = {}
    for k, name in enumerate(configs):
        procs[name] = phase1.start_training(name, gpus[k], work, media, train_deadline)
        ready = work / "sft" / f"ready_{name}"
        t_wait = time.time()
        while not ready.exists() and procs[name].poll() is None and time.time() - t_wait < 60 * 60:
            time.sleep(5)
        print(f"[exp_a] trainer {name} on GPU {gpus[k]}: {ready.read_text() if ready.exists() else 'not ready'}", flush=True)
    for name, proc in procs.items():
        proc.wait()
        print(f"[exp_a] trainer {name} exited with code {proc.returncode}", flush=True)

    summaries = {}
    for name in configs:
        path = work / "logs" / f"sft_{name}_summary.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text())
    ok = [c for c in configs if c in summaries and (work / "adapters" / c / "adapter_config.json").exists()]
    if not ok:
        raise SystemExit("No adapter was trained. See logs/sft_*_failed.json and logs/sft_*_stdout.log.")

    # ------------------------------------------------------------------ answer dev100, one adapter per GPU
    specs = {c: J.StageSpec(name=f"sft9b_{c.removesuffix('_9B')}", backend="qwen9b", prompt="p_new",
                            max_new_tokens=S.QWEN_MAX_NEW_TOKENS, adapter=f"adapters/{c}",
                            note="Qwen3.5-9B NF4 + LoRA, same prompt/pixels/cap as sft_S1/S2 (4B)") for c in ok}
    for spec in specs.values():
        J.build_plan(spec, work, eval_set)
    results, threads = {}, []

    def answer(cfg, spec, gpu):
        try:
            results[cfg] = launcher.run_stage(work, spec.name, sys.executable, [gpu], 1, hard_deadline, media)
        except Exception as exc:
            results[cfg] = {"error": f"{type(exc).__name__}: {exc}"}

    for k, (cfg, spec) in enumerate(specs.items()):
        th = threading.Thread(target=answer, args=(cfg, spec, gpus[k % len(gpus)]))
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
    merge.write_completion(work, "exp_a_completion", {
        "finished_utc": utc_now(), "hours": round(hours_since(t0), 2), "training": summaries, "answering": results,
        "status": status, "format_check": checks})
    print(f"\n[exp_a] finished in {hours_since(t0):.2f} h")
    for key, st in status.items():
        print(f"  {key:26} {st['answered']}/{st['planned']} errors={st['errors']} fallbacks={st['fallbacks']}")


if __name__ == "__main__":
    main()
