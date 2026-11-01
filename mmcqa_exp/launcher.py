"""Start worker processes for one stage, wait for them, and retry what failed.

``python_exe`` selects the Python environment (the Pangea stages run in their own venv).
Workers start one at a time: the next starts only after the previous wrote ``ready_<k>``, so peak
CPU RAM stays at one model load.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from . import jobs as J, settings as S
from .util import read_jsonl

POLL_SECONDS = 60        # tests lower this


def _tail(path, n=15) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def missing_keys(work, stage) -> list:
    paths = J.stage_paths(work, stage)
    plan = read_jsonl(paths["plan"])
    from .worker import done_keys
    done = done_keys(paths["raw"])
    return [j["key"] for j in plan if j["key"] not in done]


def _run_round(work, stage, python_exe, gpus, workers_per_gpu, deadline, media, round_no, ready_timeout, extra_env=None):
    paths = J.stage_paths(work, stage)
    paths["raw"].mkdir(parents=True, exist_ok=True)
    for stale in paths["raw"].glob("ready_*"):
        stale.unlink()
    logs = Path(work) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_workers = len(gpus) * workers_per_gpu
    procs, ready_count, handles = [], 0, []
    for k in range(n_workers):
        gpu = gpus[k % len(gpus)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
                   PYTHONUNBUFFERED="1", PYTHONPATH=str(S.REPO_ROOT))
        env.update(extra_env or {})
        log_path = logs / f"{stage}_worker_{k}.log"
        log = log_path.open("a")
        handles.append(log)
        log.write(f"\n===== round {round_no} =====\n")
        log.flush()
        proc = subprocess.Popen(
            [python_exe, "-m", "mmcqa_exp.worker", "--workdir", str(work), "--stage", stage,
             "--worker", str(k), "--workers", str(n_workers), "--media", str(media),
             "--deadline", str(deadline or 0)],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(S.REPO_ROOT))
        procs.append((proc, log_path))
        ready = paths["raw"] / f"ready_{k}"
        t0 = time.time()
        while not ready.exists() and proc.poll() is None and time.time() - t0 < ready_timeout:
            time.sleep(min(3, POLL_SECONDS))
        state = "ready" if ready.exists() else (
            f"EXITED (code {proc.poll()})" if proc.poll() is not None else "NOT READY (timeout)")
        ready_count += ready.exists()
        print(f"[launcher] {stage} worker {k} on GPU {gpu}: {state}", flush=True)
        if not ready.exists():
            print(_tail(log_path), flush=True)
    if ready_count == 0 and any(p.poll() not in (0, None) for p, _ in procs):
        for p, _ in procs:
            if p.poll() is None:
                p.kill()
        for h in handles:
            h.close()
        raise RuntimeError(f"No worker of stage '{stage}' became ready. Last log lines:\n"
                           + "\n---\n".join(_tail(lp) for _, lp in procs))
    total = len(read_jsonl(paths["plan"]))
    t_wait = time.time()
    last_print = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(POLL_SECONDS)
        if time.time() - last_print >= 60:
            last_print = time.time()
            print(f"[launcher] {stage}: {total - len(missing_keys(work, stage))}/{total} answered, "
                  f"{(time.time() - t_wait) / 60:.0f} min this round", flush=True)
    codes = [p.returncode for p, _ in procs]
    for h in handles:
        h.close()
    print(f"[launcher] {stage} round {round_no} finished, exit codes {codes}", flush=True)
    return codes


def run_stage(work, stage, python_exe, gpus, workers_per_gpu, deadline, media, rounds=2, ready_timeout=2400, extra_env=None):
    """Run ``stage`` until all jobs have an error-free answer, the deadline passes or ``rounds`` run out."""
    t0 = time.time()
    for round_no in range(1, rounds + 1):
        missing = missing_keys(work, stage)
        if not missing:
            break
        if deadline and time.time() > deadline:
            print(f"[launcher] {stage}: deadline passed with {len(missing)} items missing", flush=True)
            break
        print(f"[launcher] {stage} round {round_no}: {len(missing)} items to answer on GPUs {list(gpus)}", flush=True)
        _run_round(work, stage, python_exe, list(gpus), workers_per_gpu, deadline, media, round_no, ready_timeout, extra_env)
    wall = time.time() - t0
    left = len(missing_keys(work, stage))
    total = len(read_jsonl(J.stage_paths(work, stage)["plan"]))
    return {"wall_seconds": round(wall, 1), "planned": total, "missing": left, "answered": total - left}


def list_gpu_indices() -> list:
    from .util import gpu_names
    return list(range(len(gpu_names())))
