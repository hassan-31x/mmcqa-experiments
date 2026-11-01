"""Phase 2 orchestrator: every candidate answer the Pangea + Qwen ensembles need, on dev500.

    python -m mmcqa_exp.phase2 --workdir /kaggle/working/results --budget-hours 11
    python -m mmcqa_exp.phase2 --workdir /kaggle/working/results --limit 4            # smoke run
    python -m mmcqa_exp.phase2 --workdir /kaggle/working/results --stages fusion_qwen # rerun one stage

Stages run in priority order. A stage is skipped (and the reason recorded) if a dependency is incomplete
or if it cannot finish inside the budget. Pangea runs in its own venv (transformers 4.37), Qwen in the
main environment (transformers 5.x).
"""

import argparse
import sys
import time
from pathlib import Path

from . import data, jobs as J, launcher, merge, settings as S
from .util import dump_json, environment, hours_since, utc_now


def stage_specs(samples: bool) -> list:
    specs = [
        J.StageSpec("pangea_p0", "pangea", "p0", S.PANGEA_MAX_NEW_TOKENS,
                    note="reproduces the earlier Pangea run (old prompt, 128 tokens)"),
        J.StageSpec("qwen_z1", "qwen", "p_new", S.QWEN_MAX_NEW_TOKENS,
                    note="zero-shot control for Phase 1: same prompt, pixels and cap as the fine-tuned model"),
        J.StageSpec("cascade_qwen2pangea", "pangea", "p_cascade", S.PANGEA_MAX_NEW_TOKENS, deps=("qwen_z1",)),
        J.StageSpec("fusion_qwen", "qwen", "p_fusion", S.QWEN_MAX_NEW_TOKENS, deps=("pangea_p0", "qwen_z1")),
    ]
    if samples:
        for k in range(2):
            specs.append(J.StageSpec(f"qwen_z1_s{k}", "qwen", "p_new", S.QWEN_MAX_NEW_TOKENS, do_sample=True,
                                     temperature=0.7, top_p=0.9, seed=k, note="sampled candidate for MBR"))
    return specs


def initial_rates(gpus: int) -> dict:
    """Items per second per backend, before anything has been measured."""
    out = {}
    for backend, spw in S.DEFAULT_SEC_PER_ITEM.items():
        wpg = S.WORKERS_PER_GPU[backend]
        eff = S.SCALING_EFFICIENCY[backend] if wpg > 1 else 1.0
        out[backend] = gpus * wpg * eff / spw
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/kaggle/working/results"))
    ap.add_argument("--media", type=Path, default=None)
    ap.add_argument("--envs", type=Path, default=None, help="where the Pangea venv goes (default: next to the media cache, outside /kaggle/working)")
    ap.add_argument("--budget-hours", type=float, default=S.DEFAULT_BUDGET_HOURS)
    ap.add_argument("--stages", nargs="*", default=None, help="only these stages (dependencies must already be merged)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: items per track")
    ap.add_argument("--samples", action="store_true", help="also generate two sampled Qwen candidates for MBR")
    ap.add_argument("--devtest", action="store_true", help="after the dev stages, answer devtest with pangea_p0 and qwen_z1")
    ap.add_argument("--pangea-workers", type=int, default=None, help="override workers per GPU for Pangea")
    args = ap.parse_args(argv)

    t0 = time.time()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    media = args.media or S.kaggle_media_dir()
    envs = args.envs or (media.parent / "mmcqa_envs")
    hard_deadline = t0 + args.budget_hours * 3600 - S.FINAL_RESERVE_MINUTES * 60
    gpus = launcher.list_gpu_indices()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    if args.pangea_workers:
        S.WORKERS_PER_GPU["pangea"] = args.pangea_workers
    specs = [s for s in stage_specs(args.samples) if not args.stages or s.name in args.stages]
    if not specs:
        raise SystemExit(f"No stage selected. Available: {[s.name for s in stage_specs(True)]}")

    eval_set = data.eval_rows()
    if args.limit:
        eval_set = {t: rows[:args.limit] for t, rows in eval_set.items()}
    data.fetch_images("dev", media, data.images_needed(eval_set))
    data.official_repo(work)

    pangea_py, pangea_error = None, None
    if any(s.backend == "pangea" for s in specs):
        from .env_setup import setup_pangea_env
        try:
            pangea_py = setup_pangea_env(envs / "pangea")
        except Exception as exc:
            pangea_error = f"{type(exc).__name__}: {exc}"
            print("[phase2] PANGEA ENVIRONMENT FAILED; Pangea stages will be skipped:\n" + pangea_error, flush=True)
    dump_json(work / "logs" / "phase2_run_config.json", {
        "started_utc": utc_now(), "args": {k: str(v) for k, v in vars(args).items()},
        "dataset_revision": S.DATASET_REVISION, "qwen": [S.QWEN_REPO, S.QWEN_REVISION],
        "pangea": [S.PANGEA_REPO, S.PANGEA_REVISION], "qwen_max_pixels": S.QWEN_MAX_PIXELS,
        "workers_per_gpu": S.WORKERS_PER_GPU, "eval_rows_per_track": len(eval_set["en"]),
        "environment": environment({"pangea": pangea_py} if pangea_py else None), "pangea_env_error": pangea_error,
        "stages": [s.describe() for s in specs]})

    rates = initial_rates(len(gpus))
    report, merged_ok = {}, {}
    n_items = sum(len(r) for r in eval_set.values())

    def run_one(spec, rows, split_label):
        for dep in spec.deps:
            ok = merged_ok.get(dep)
            if ok is None:                               # not run in this session: look on disk
                ok = all(len(list((work / "predictions" / dep).glob(f"{spec.split}_{t}.jsonl"))) == 1 for t in S.TRACKS)
            if not ok:
                return {"state": "skipped", "reason": f"dependency '{dep}' is not complete"}
        if spec.backend == "pangea" and pangea_py is None:
            return {"state": "skipped", "reason": f"Pangea environment unavailable: {pangea_error}"}
        n = sum(len(r) for r in rows.values())
        est_h = n / rates[spec.backend] / 3600
        left_h = (hard_deadline - time.time()) / 3600
        if est_h > left_h:
            return {"state": "skipped", "reason": f"estimated {est_h:.2f} h > {left_h:.2f} h left"}
        J.build_plan(spec, work, rows)
        wpg = S.WORKERS_PER_GPU[spec.backend]
        exe = pangea_py if spec.backend == "pangea" else sys.executable
        print(f"\n[phase2] === {spec.name} ({spec.backend}, {n} items, est {est_h:.2f} h, {left_h:.2f} h left) ===", flush=True)
        res = launcher.run_stage(work, spec.name, exe, gpus, wpg, hard_deadline, media)
        if res["answered"] > 0 and res["wall_seconds"] > 60:
            rates[spec.backend] = res["answered"] / res["wall_seconds"]
        st = merge.merge_stage(work, spec.name)
        chk = merge.check_stage_format(work, spec.name, st)
        complete = all(v["answered"] == v["planned"] for v in st.values())
        merged_ok[spec.name] = complete
        return {"state": "done" if complete else "incomplete", "run": res, "status": st, "format_check": chk,
                "items_per_second": round(rates[spec.backend], 4)}

    for spec in specs:
        try:
            report[spec.name] = run_one(spec, eval_set, "dev")
        except Exception as exc:                          # one broken stage must not lose the others
            report[spec.name] = {"state": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            print(f"[phase2] stage {spec.name} FAILED: {exc}", flush=True)
        merge.write_completion(work, "phase2_completion", {"hours": round(hours_since(t0), 2), "stages": report})

    if args.devtest and not args.limit:
        dev_rows = {t: data.load_rows("devtest", t) for t in S.TRACKS}
        data.fetch_images("devtest", media, data.images_needed(dev_rows))
        for base in ("qwen_z1", "pangea_p0"):
            src = next((s for s in stage_specs(False) if s.name == base), None)
            if src is None or (args.stages and base not in args.stages):
                continue
            spec = J.StageSpec(f"devtest_{base}", src.backend, src.prompt, src.max_new_tokens, split="devtest")
            try:
                report[spec.name] = run_one(spec, dev_rows, "devtest")
            except Exception as exc:
                report[spec.name] = {"state": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    merge.write_completion(work, "phase2_completion", {"hours": round(hours_since(t0), 2), "stages": report})
    print(f"\n[phase2] finished in {hours_since(t0):.2f} h")
    for name, r in report.items():
        extra = r.get("reason") or ", ".join(f"{k.split('|')[1]}:{v['answered']}/{v['planned']}"
                                              for k, v in r.get("status", {}).items())
        print(f"  {name:24} {r['state']:10} {extra}")


if __name__ == "__main__":
    main()
