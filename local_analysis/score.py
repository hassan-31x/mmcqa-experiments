"""Score systems with the organisers' scorer (unmodified). Run with the pinned scorer environment:

    ../external/.venv-scorer/bin/python -m local_analysis.score --runs results/phase1/TAG results/phase2/TAG \
        --out results/analysis/NAME [--expect-rows 500] [--raw]

Per system and track it calls ``backbone.evaluate`` and also computes per-item P/R/F1 with the same scorer
object, asserting that the per-item mean equals the official F1. Systems already present in the output are
skipped (incremental), unless --rescore.
"""

import argparse
import os
import sys
from pathlib import Path

from .common import TRACKS, discover_predictions, eval_ids, gold_rows, json_dump, read_jsonl, write_jsonl
from .paths import HF_SNAPSHOTS, OFFICIAL, REPO


def load_backbone():
    sys.path.insert(0, str(OFFICIAL / "scorer"))
    import backbone
    os.environ.setdefault("MMCQA_MODEL_DIR", str(HF_SNAPSHOTS / backbone.HUB_REVISION))
    return backbone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], type=Path, help="pulled run folders (contain results/predictions)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect-rows", type=int, default=500)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--raw", action="store_true", help="also score the uncleaned raw_answer as <system>__raw")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="score only ids a system has (default: fail)")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    backbone = load_backbone()
    gold = gold_rows()
    ids = eval_ids(args.expect_rows)
    gold_by = {t: {r["id"]: r for r in gold[t]} for t in TRACKS}
    preds, origin = discover_predictions([*args.runs, out])

    items_path, metrics_path = out / "per_item_scores.jsonl", out / "official_metrics.json"
    items = [] if args.rescore else read_jsonl(items_path)
    done = {r["system"] for r in items}
    import json
    metrics = {} if args.rescore or not metrics_path.exists() else json.loads(metrics_path.read_text())["results"]

    todo = []
    for system in sorted(preds):
        if origin[system] == "selection":
            continue                                          # derived from candidates; scored by ensemble_select
        if args.systems and system not in args.systems:
            continue
        todo.append((system, "answer"))
        if args.raw:
            todo.append((f"{system}__raw", "raw_answer"))
    todo = [(s, f) for s, f in todo if s not in done]
    print(f"[score] {len(todo)} systems to score: {[s for s, _ in todo]}", flush=True)

    rouge = backbone.get_rouge()
    for system, field in todo:
        base = system.removesuffix("__raw")
        for t in TRACKS:
            have = preds[base].get(t, {})
            missing = [i for i in ids if i not in have]
            if missing and not args.allow_partial:
                raise SystemExit(f"{base}/{t}: {len(missing)} of {len(ids)} ids missing (e.g. {missing[0][:12]}). "
                                 "A missing item scores 0 on the leaderboard; finish the run or pass --allow-partial.")
            use = [i for i in ids if i in have]
            if not use:
                continue                                      # e.g. a pivot system that only covers 3 tracks
            gold_path = out / "files" / f"gold_{t}_{len(use)}.jsonl"
            write_jsonl(gold_path, ({"id": i, "answer": {t: gold_by[t][i]["answer"]}} for i in use))
            pred_path = out / "files" / f"{system}_{t}.jsonl"
            write_jsonl(pred_path, ({"id": i, "answer": have[i].get(field, "")} for i in use))
            result = backbone.evaluate(str(gold_path), str(pred_path), "dev",
                                       submission_metadata=[{"name": "Language", "value": t}])
            block = result["submission_result"]
            metrics[f"{system}|{t}"] = {**block, "n_items": result["submission_metadata"]["n_items"],
                                        "n_empty": result["submission_metadata"]["n_empty"]}
            refs = [backbone._clean(gold_by[t][i]["answer"]) for i in use]
            cands = [backbone._clean(have[i].get(field, "")) for i in use]
            nz = [k for k, c in enumerate(cands) if c]
            P, R, F = ([0.0] * len(use) for _ in range(3))
            if nz:
                p, r, f = backbone.get_scorer().score([cands[k] for k in nz], [refs[k] for k in nz],
                                                      verbose=False, batch_size=backbone.BATCH_SIZE)
                for j, k in enumerate(nz):
                    P[k], R[k], F[k] = float(p[j]), float(r[j]), float(f[j])
            mean_f = sum(F) / len(F)
            assert abs(mean_f - block["BERTScore-F1"]) < 1e-4, (system, t, mean_f, block["BERTScore-F1"])
            for k, i in enumerate(use):
                rec = have[i]
                items.append({"system": system, "track": t, "id": i, "question": rec.get("question") or gold_by[t][i]["question"],
                              "reference": refs[k], "prediction": cands[k], "mm_p": P[k], "mm_r": R[k], "mm_f1": F[k],
                              "rougeL": rouge.score(refs[k], cands[k])["rougeL"].fmeasure if cands[k] else 0.0,
                              "words": len(cands[k].split()), "fallback_used": rec.get("fallback_used"),
                              "latency_seconds": rec.get("latency_seconds"), "mean_logprob": rec.get("mean_logprob"),
                              "new_tokens": rec.get("new_tokens"), "hit_max_new_tokens": rec.get("hit_max_new_tokens")})
            print(f"  {system:28} {t:4} n={len(use):5} F1={block['BERTScore-F1']:.4f} P={block['BERTScore-P']:.4f} "
                  f"R={block['BERTScore-R']:.4f}", flush=True)
        write_jsonl(items_path, items)                      # checkpoint after each system
        json_dump(metrics_path, {"scorer": {"model": backbone.HUB_MODEL, "revision": backbone.HUB_REVISION,
                                            "layer": backbone.NUM_LAYERS}, "expect_rows": args.expect_rows,
                                 "results": metrics})
    print(f"[score] wrote {items_path}")


if __name__ == "__main__":
    main()
