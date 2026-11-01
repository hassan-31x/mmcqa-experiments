"""Merge worker files into per-system prediction files and official-format submission files."""

import json
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

from . import jobs as J, settings as S
from .util import read_jsonl, read_jsonl_many, write_jsonl


def latest_answers(raw_dir) -> dict:
    """Last record per key; an error-free record always beats an errored one."""
    best = {}
    for row in read_jsonl_many(raw_dir):
        old = best.get(row["key"])
        if old is None or not row.get("error") or old.get("error"):
            best[row["key"]] = row
    return best


def merge_stage(work, stage) -> dict:
    """Write ``predictions/<stage>/<split>_<track>.jsonl`` and ``submission/<stage>/<split>/...``."""
    work = Path(work)
    paths = J.stage_paths(work, stage)
    plan = read_jsonl(paths["plan"])
    answers = latest_answers(paths["raw"])
    status = {}
    for track in S.TRACKS:
        jobs = [j for j in plan if j["track"] == track]
        if not jobs:
            continue
        split = jobs[0]["split"]
        got = [(j, answers[j["key"]]) for j in jobs if j["key"] in answers]
        write_jsonl(paths["pred"] / f"{split}_{track}.jsonl", (
            {"system": stage, "track": track, "split": split, "id": j["id"], "question": j["question"],
             "answer": a["answer"], "raw_answer": a["raw_answer"], "fallback_used": a["fallback_used"],
             "error": a["error"], "latency_seconds": a["latency_seconds"], "new_tokens": a["new_tokens"],
             "prompt_tokens": a["prompt_tokens"], "hit_max_new_tokens": a["hit_max_new_tokens"],
             "mean_logprob": a["mean_logprob"], "peak_mem_gb": a["peak_mem_gb"],
             "prompt": j["prompt"], "meta": j["meta"]} for j, a in got))
        write_jsonl(paths["submission"] / split / f"prediction_qa_{track}.jsonl",
                    ({"id": j["id"], "answer": a["answer"], "lang": track} for j, a in got))
        lat = [a["latency_seconds"] for _, a in got if a["latency_seconds"] is not None]
        status[f"{stage}|{track}"] = {
            "planned": len(jobs), "answered": len(got),
            "errors": sum(1 for _, a in got if a["error"]),
            "fallbacks": sum(1 for _, a in got if a["fallback_used"]),
            "hit_cap": sum(1 for _, a in got if a["hit_max_new_tokens"]),
            "mean_latency": round(statistics.mean(lat), 2) if lat else None,
            "peak_mem_gb": max([a["peak_mem_gb"] for _, a in got if a["peak_mem_gb"] is not None] or [None])}
    return status


def check_stage_format(work, stage, status) -> dict:
    """Run the organisers' format checker on every complete file; zip devtest files for upload."""
    work = Path(work)
    checker = work / "mmcultureqa" / "format_checker" / "check_format.py"
    report = {}
    for key, st in status.items():
        _, track = key.split("|")
        plan = [j for j in read_jsonl(J.stage_paths(work, stage)["plan"]) if j["track"] == track]
        split = plan[0]["split"]
        pred = J.stage_paths(work, stage)["submission"] / split / f"prediction_qa_{track}.jsonl"
        if st["answered"] != st["planned"]:
            report[key] = f"incomplete ({st['answered']}/{st['planned']}), not checked"
            continue
        if not checker.exists():
            report[key] = "format checker not available (starter kit not cloned)"
            continue
        ids = work / "ids" / f"{stage}_{track}_ids.jsonl"
        write_jsonl(ids, ({"id": j["id"]} for j in plan))
        res = subprocess.run([sys.executable, str(checker), "--pred", str(pred), "--lang", track, "--ids", str(ids)],
                             capture_output=True, text=True)
        detail = " ".join((res.stdout + res.stderr).split())[-300:]
        report[key] = ("OK" if res.returncode == 0 else "FAILED") + " | " + detail
        if split == "devtest" and res.returncode == 0:
            with zipfile.ZipFile(pred.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(pred, "prediction.jsonl")
    return report


def write_completion(work, name, payload) -> None:
    path = Path(work) / "logs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
