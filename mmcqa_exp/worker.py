"""One inference worker: loads one backend on one GPU and answers its share of a stage's plan.

    CUDA_VISIBLE_DEVICES=0 python -m mmcqa_exp.worker --workdir W --stage qwen_z1 \
        --worker 0 --workers 4 --media /kaggle/temp/mmcqa_media --deadline 1760000000

Every answer is appended and flushed at once. Items already answered without error by any worker
are skipped, so a restarted stage continues where it stopped. Exit codes: 3 = repeated CUDA OOM,
4 = repeated failures of any kind (something systematic is broken).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import jobs as J
from .postprocess import finalize
from .util import read_jsonl, read_jsonl_many

MAX_CONSECUTIVE_OOM = 3
MAX_CONSECUTIVE_ERRORS = 12


def done_keys(raw_dir) -> set:
    return {r["key"] for r in read_jsonl_many(raw_dir) if not r.get("error")}


def stable_seed(key: str, seed) -> int:
    return (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) ^ int(seed or 0)) % (2 ** 31)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--deadline", type=float, default=0, help="unix time after which no new item is started")
    args = ap.parse_args(argv)

    paths = J.stage_paths(args.workdir, args.stage)
    raw_dir = paths["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = read_jsonl(paths["plan"])
    if not plan:
        sys.exit(f"Empty or missing plan: {paths['plan']}")
    mine = [job for k, job in enumerate(plan) if k % args.workers == args.worker]
    skip = done_keys(raw_dir)
    todo = [job for job in mine if job["key"] not in skip]
    print(f"[worker {args.worker}] {args.stage}: {len(mine)} jobs, {len(todo)} to do", flush=True)
    ready = raw_dir / f"ready_{args.worker}"
    if not todo:
        ready.write_text(json.dumps({"load_seconds": 0}))
        print(f"[worker {args.worker}] nothing to do", flush=True)
        return 0

    import torch
    from PIL import Image
    from .backends import get_backend

    first = todo[0]
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:
        tf_version = "not installed"
    print(f"[worker {args.worker}] python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"transformers {tf_version}", flush=True)
    adapter = str(args.workdir / first["adapter"]) if first.get("adapter") else None
    t_load = time.time()
    backend = get_backend(first["backend"])(adapter=adapter)
    backend.load()
    ready.write_text(json.dumps({"load_seconds": round(time.time() - t_load, 1)}))
    print(f"[worker {args.worker}] model ready in {time.time() - t_load:.0f}s", flush=True)

    out_path = raw_dir / f"worker_{args.worker}.jsonl"
    consecutive_oom = consecutive_err = 0
    started = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        if out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
            fh.write("\n")                       # a previous run was killed mid-line
        for n, job in enumerate(todo, 1):
            if args.deadline and time.time() > args.deadline:
                print(f"[worker {args.worker}] time budget reached after {n - 1} items", flush=True)
                break
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            res, error, oom = {}, None, False
            try:
                with Image.open(args.media / job["image"]) as source:
                    image = source.convert("RGB")
                if job["do_sample"]:
                    torch.manual_seed(stable_seed(job["key"], job.get("seed")))
                res = backend.answer(image, job["prompt"], job["max_new_tokens"],
                                     do_sample=job["do_sample"], temperature=job.get("temperature"),
                                     top_p=job.get("top_p"))
            except Exception as exc:             # a failed item gets the non-empty fallback, never ""
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                oom = type(exc).__name__ == "OutOfMemoryError"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raw_answer = res.get("raw_answer", "")
            answer, fallback = finalize(raw_answer, job["question"], hit_cap=bool(res.get("hit_max_new_tokens")))
            fh.write(json.dumps({
                "key": job["key"], "stage": job["stage"], "system": job["system"], "split": job["split"],
                "track": job["track"], "id": job["id"], "raw_answer": raw_answer, "answer": answer,
                "fallback_used": fallback, "error": error,
                "latency_seconds": round(time.time() - t0, 3), "new_tokens": res.get("new_tokens"),
                "prompt_tokens": res.get("prompt_tokens"),
                "hit_max_new_tokens": res.get("hit_max_new_tokens"), "mean_logprob": res.get("mean_logprob"),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
                "worker": args.worker, "finished_unix": round(time.time(), 1)}, ensure_ascii=False) + "\n")
            fh.flush()
            consecutive_oom = consecutive_oom + 1 if oom else 0
            consecutive_err = consecutive_err + 1 if error else 0
            if error:
                print(f"[worker {args.worker}] ERROR {job['key']}: {error}", flush=True)
            if consecutive_oom >= MAX_CONSECUTIVE_OOM:
                print(f"[worker {args.worker}] {consecutive_oom} CUDA OOMs in a row: exiting", flush=True)
                return 3
            if consecutive_err >= MAX_CONSECUTIVE_ERRORS:
                print(f"[worker {args.worker}] {consecutive_err} errors in a row: exiting", flush=True)
                return 4
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - started)
                print(f"[worker {args.worker}] {n}/{len(todo)} {rate:.3f} items/s "
                      f"eta {(len(todo) - n) / rate / 60:.0f} min | {job['stage']} {job['track']}", flush=True)
    print(f"[worker {args.worker}] finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
