"""Pairwise agreement between candidate answers (mmBERT F1), the input to MBR selection and the router.

    ../external/.venv-scorer/bin/python -m local_analysis.mbr --runs RUN... --out results/analysis/NAME \
        --pool pangea_p0 qwen_z1 sft_S1 tfidf_retrieval ...

F1 is symmetric, so only unordered pairs are scored. Results are cached in pairwise_sim.jsonl and reused.
CPU cost: roughly 30-60 minutes for 8 candidates x 2,000 items.
"""

import argparse
import itertools
from pathlib import Path

from .common import TRACKS, discover_predictions, eval_ids, read_jsonl
from .score import load_backbone


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--expect-rows", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args(argv)
    backbone = load_backbone()
    preds, _ = discover_predictions([*args.runs, args.out])
    missing = [s for s in args.pool if s not in preds]
    if missing:
        raise SystemExit(f"Unknown systems in --pool: {missing}. Available: {sorted(preds)}")
    ids = eval_ids(args.expect_rows)
    path = args.out / "pairwise_sim.jsonl"
    cache = {(r["track"], r["id"], r["a"], r["b"]) for r in read_jsonl(path)}
    todo = []
    for t in TRACKS:
        for i in ids:
            for a, b in itertools.combinations(sorted(args.pool), 2):
                if (t, i, a, b) in cache or i not in preds[a].get(t, {}) or i not in preds[b].get(t, {}):
                    continue
                todo.append((t, i, a, b, backbone._clean(preds[a][t][i]["answer"]), backbone._clean(preds[b][t][i]["answer"])))
    print(f"[mbr] {len(todo)} pairs to score ({len(cache)} cached)", flush=True)
    import json
    with path.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s:s + args.chunk]
            nz = [k for k, c in enumerate(chunk) if c[4] and c[5]]
            f1 = [0.0] * len(chunk)
            if nz:
                _, _, f = backbone.get_scorer().score([chunk[k][4] for k in nz], [chunk[k][5] for k in nz],
                                                      verbose=False, batch_size=backbone.BATCH_SIZE)
                for j, k in enumerate(nz):
                    f1[k] = float(f[j])
            for c, v in zip(chunk, f1):
                fh.write(json.dumps({"track": c[0], "id": c[1], "a": c[2], "b": c[3], "f1": v}) + "\n")
            fh.flush()
            print(f"[mbr] {min(s + args.chunk, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
