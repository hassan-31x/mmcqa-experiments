"""Convert an earlier run (local/runs/<run>/predictions.jsonl, dev100) into the candidate format used here.

    python -m local_analysis.import_old_run --run ../local/runs/culturalpangea_7b_4e2a96ac07a9 \
        --name old_pangea --out results/analysis/NAME

The answer is stored exactly as submitted back then, so scoring it reproduces the historical number.
It only covers the 100 ids per track those runs answered (score with --expect-rows 100 or --allow-partial).
"""

import argparse
from pathlib import Path

from .common import TRACKS, read_jsonl, write_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    by_track = {t: [] for t in TRACKS}
    for r in read_jsonl(args.run / "predictions.jsonl"):
        if r.get("task", "qa") != "qa" or r["split"] != "dev" or r["variant"] != "original" or r["status"] != "ok":
            continue
        track = r["track"].rsplit("_", 1)[-1]
        by_track[track].append({"system": args.name, "track": track, "split": "dev", "id": r["id"],
                                "question": r["question"], "answer": r["prediction"], "raw_answer": r["prediction"],
                                "fallback_used": False, "error": None, "latency_seconds": r.get("latency_seconds"),
                                "mean_logprob": None})
    for t, rows in by_track.items():
        write_jsonl(args.out / "candidates" / args.name / f"dev_{t}.jsonl", rows)
    print(f"{args.name}: " + ", ".join(f"{t}={len(r)}" for t, r in by_track.items()))


if __name__ == "__main__":
    main()
