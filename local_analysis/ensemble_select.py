"""Selection ensembles: each method picks, per item, one of the candidate answers. No GPU needed.

    python -m local_analysis.ensemble_select --out results/analysis/NAME [--expect-rows 500] [--pool NAME=a,b,c ...]

Methods (the oracle is an upper bound, never a submission):
    oracle              best candidate per item, using the gold F1
    per_track_best      the best single system on each track (fit on all items: optimistic)
    per_track_best-cv   same, fit on the training folds only
    mbr                 candidate with the highest mean agreement with the other candidates (needs >= 3 candidates)
    mbr_weighted-cv     agreement weighted by each system's training-fold F1
    router-cv           gradient-boosted regressor predicts each candidate's F1 (5-fold GroupKFold by id)
Fit-based methods are reported cross-validated; GroupKFold keeps all four tracks of an id in one fold.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from .common import (TRACKS, discover_predictions, eval_ids, load_per_item, macro_ci, per_track_means, read_jsonl,
                     write_jsonl)

N_FOLDS = 5


# ----------------------------------------------------------------------------- core (pure numpy, unit-tested)
def folds_by_group(groups: np.ndarray, n_splits: int = N_FOLDS):
    from sklearn.model_selection import GroupKFold
    n_splits = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def choose_oracle(F):
    return F.argmax(axis=1)


def fit_track_best(F, track_idx):
    return np.array([F[track_idx == j].mean(axis=0).argmax() for j in range(len(TRACKS))])


def choose_track_best(best, track_idx):
    return best[track_idx]


def choose_mbr(SIM, weights=None):
    """``SIM[n, c, j]`` = agreement between candidates c and j on item n (NaN on the diagonal)."""
    n, s, _ = SIM.shape
    w = np.ones(s) if weights is None else np.asarray(weights, dtype=float)
    scores = np.zeros((n, s))
    for c in range(s):
        others = [j for j in range(s) if j != c]
        vals, ws = SIM[:, c, others], w[others]
        ok = ~np.isnan(vals)
        scores[:, c] = (np.where(ok, vals, 0) * ws).sum(axis=1) / np.maximum((ok * ws).sum(axis=1), 1e-9)
    return scores.argmax(axis=1)


def cv_choose(F, groups, track_idx, SIM=None, X=None, method="per_track_best"):
    """Cross-validated choice for the fit-based methods. ``X`` is the router's (N, S, d) feature tensor."""
    n, s = F.shape
    chosen = np.zeros(n, dtype=int)
    for tr, te in folds_by_group(groups):
        if method == "per_track_best":
            chosen[te] = choose_track_best(fit_track_best(F[tr], track_idx[tr]), track_idx[te])
        elif method == "mbr_weighted":
            chosen[te] = choose_mbr(SIM[te], F[tr].mean(axis=0))
        elif method == "router":
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
            Xtr, Xte = X[tr].reshape(len(tr) * s, -1), X[te].reshape(len(te) * s, -1)
            usable = ~np.isnan(Xtr).all(axis=0)              # e.g. log-probs when no candidate has them
            model.fit(Xtr[:, usable], F[tr].reshape(-1))
            chosen[te] = model.predict(Xte[:, usable]).reshape(len(te), s).argmax(axis=1)
        else:
            raise ValueError(method)
    return chosen


# ----------------------------------------------------------------------------- feature construction
def arabic_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters) if letters else 0.0


def build_features(systems, track_idx, words, qwords, logprob, fallback, script_bad, SIM):
    n, s = words.shape
    tr = np.eye(len(TRACKS))[track_idx]                              # (n, 4)
    cols = []
    for c in range(s):
        sysoh = np.zeros((n, s)); sysoh[:, c] = 1
        others = [j for j in range(s) if j != c]
        mean_sim = np.nanmean(SIM[:, c, others], axis=1) if others else np.full(n, np.nan)
        cols.append(np.column_stack([tr, sysoh, words[:, c], words[:, c] / np.maximum(qwords, 1), logprob[:, c],
                                     fallback[:, c], script_bad[:, c], mean_sim, SIM[:, c, :]]))
    return np.stack(cols, axis=1)                                    # (n, s, d)


# ----------------------------------------------------------------------------- data assembly
def assemble(df, preds, pool, sim_rows, ids):
    """Return aligned arrays for the items on which every pool system has an answer on all four tracks."""
    sub = df[df.system.isin(pool)]
    have = {s: set(zip(sub[sub.system == s].track, sub[sub.system == s].id)) for s in pool}
    keep = [i for i in ids if all((t, i) in have[s] for s in pool for t in TRACKS)]
    n_ids = len(keep)
    n = n_ids * len(TRACKS)
    pos = {(t, i): k * len(TRACKS) + j for k, i in enumerate(keep) for j, t in enumerate(TRACKS)}
    s_idx = {s: c for c, s in enumerate(pool)}
    S = len(pool)
    F, words, logprob, fallback, script_bad = (np.full((n, S), np.nan) for _ in range(5))
    qwords = np.zeros(n)
    text = {}
    for row in sub.itertuples():
        k = pos.get((row.track, row.id))
        if k is None:
            continue
        c = s_idx[row.system]
        F[k, c], words[k, c] = row.mm_f1, row.words
        logprob[k, c] = np.nan if row.mean_logprob is None or row.mean_logprob != row.mean_logprob else row.mean_logprob
        fallback[k, c] = 1.0 if row.fallback_used else 0.0
        frac = arabic_fraction(row.prediction)
        script_bad[k, c] = float((row.track == "en" and frac > 0.5) or (row.track != "en" and frac < 0.5))
        qwords[k] = len(str(row.question).split())
        text[(row.track, row.id, row.system)] = row.prediction
    SIM = np.full((n, S, S), np.nan)
    for r in sim_rows:
        k = pos.get((r["track"], r["id"]))
        if k is None or r["a"] not in s_idx or r["b"] not in s_idx:
            continue
        a, b = s_idx[r["a"]], s_idx[r["b"]]
        SIM[k, a, b] = SIM[k, b, a] = r["f1"]
    track_idx = np.tile(np.arange(len(TRACKS)), n_ids)
    groups = np.repeat(np.arange(n_ids), len(TRACKS))
    return {"keep": keep, "F": F, "SIM": SIM, "track_idx": track_idx, "groups": groups, "words": words, "qwords": qwords,
            "logprob": logprob, "fallback": fallback, "script_bad": script_bad, "pos": pos}


def default_pools(df, ids):
    systems = sorted(s for s in df.system.unique() if not (s.startswith(("sel_", "old_")) or s.endswith("__raw")))
    full = {s for s in systems if len(df[(df.system == s)]) >= len(ids) * len(TRACKS)}
    sft = sorted((s for s in full if s.startswith("sft_")),
                 key=lambda s: -df[df.system == s].mm_f1.mean())
    zs = [s for s in sorted(full) if not s.startswith(("sft_", "devtest_"))]
    pools = {}
    if {"pangea_p0", "qwen_z1"} <= full:
        pools["P2"] = ["pangea_p0", "qwen_z1"]
        if "tfidf_retrieval" in full:
            pools["P3"] = ["pangea_p0", "qwen_z1", "tfidf_retrieval"]
        pools["Pzs"] = zs
    if sft and "pangea_p0" in full:
        pools["Pft2"] = ["pangea_p0", sft[0]]
        if "qwen_z1" in full:
            pools["Pft3"] = ["pangea_p0", "qwen_z1", sft[0]]
        pools["Pftall"] = sorted(set(zs) | {sft[0]})
    return {k: v for k, v in pools.items() if len(v) >= 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", nargs="*", default=[], type=Path)
    ap.add_argument("--expect-rows", type=int, default=500)
    ap.add_argument("--pool", nargs="*", default=None, help="NAME=sys1,sys2,... (default: built from what exists)")
    args = ap.parse_args(argv)
    out = args.out
    df = load_per_item(out / "per_item_scores.jsonl")
    if df.empty:
        raise SystemExit("per_item_scores.jsonl is empty: run local_analysis.score first.")
    ids = eval_ids(args.expect_rows)
    preds, _ = discover_predictions([*args.runs, out])
    sim_rows = read_jsonl(out / "pairwise_sim.jsonl")
    pools = ({k: v.split(",") for k, v in (p.split("=") for p in args.pool)} if args.pool else default_pools(df, ids))
    print(f"[select] pools: {pools}", flush=True)

    summary, sel_scores = [], []
    for pname, pool in pools.items():
        d = assemble(df, preds, pool, sim_rows, ids)
        n_ids = len(d["keep"])
        if n_ids < 20:
            print(f"[select] pool {pname}: only {n_ids} complete ids, skipped", flush=True)
            continue
        F, SIM, tr, grp = d["F"], d["SIM"], d["track_idx"], d["groups"]
        sim_ok = bool((~np.isnan(SIM[:, 0, 1:])).any()) and (np.isnan(SIM).sum() / SIM.size < 0.01 + 1.0 / len(pool))
        X = build_features(pool, tr, d["words"], d["qwords"], d["logprob"], d["fallback"], d["script_bad"], SIM)
        choices = {"oracle": choose_oracle(F),
                   "per_track_best": choose_track_best(fit_track_best(F, tr), tr),
                   "per_track_best-cv": cv_choose(F, grp, tr, method="per_track_best"),
                   "router-cv": cv_choose(F, grp, tr, SIM=SIM, X=X, method="router")}
        if len(pool) >= 3 and sim_ok:
            choices["mbr"] = choose_mbr(SIM)
            choices["mbr_weighted-cv"] = cv_choose(F, grp, tr, SIM=SIM, method="mbr_weighted")
        elif len(pool) >= 3:
            print(f"[select] pool {pname}: pairwise similarities missing, MBR skipped (run local_analysis.mbr)", flush=True)
        for s in pool:                                              # single systems for reference, same items
            c = pool.index(s)
            summary.append(row(f"single:{s}", pname, F[:, c], n_ids, pool, np.full(len(F), c)))
        for method, idx in choices.items():
            name = f"sel_{method}_{pname}"
            chosen_F = F[np.arange(len(F)), idx]
            summary.append(row(method, pname, chosen_F, n_ids, pool, idx))
            if method == "oracle":
                continue                                            # an upper bound is not a system to judge or submit
            for t_i, t in enumerate(TRACKS):
                rows = []
                for k, i in enumerate(d["keep"]):
                    n = k * len(TRACKS) + t_i
                    src = pool[idx[n]]
                    rec = preds[src][t][i]
                    rows.append({"system": name, "track": t, "split": "dev", "id": i, "question": rec.get("question"),
                                 "answer": rec["answer"], "raw_answer": rec.get("raw_answer"), "chosen_from": src,
                                 "fallback_used": rec.get("fallback_used"), "error": None})
                write_jsonl(out / "selection" / name / f"dev_{t}.jsonl", rows)
            src_rows = df[df.system.isin(pool)].set_index(["system", "track", "id"])
            for t_i, t in enumerate(TRACKS):
                for k, i in enumerate(d["keep"]):
                    src = pool[idx[k * len(TRACKS) + t_i]]
                    r = src_rows.loc[(src, t, i)]
                    sel_scores.append({"system": name, "track": t, "id": i, "question": r.question, "reference": r.reference,
                                       "prediction": r.prediction, "mm_p": r.mm_p, "mm_r": r.mm_r, "mm_f1": r.mm_f1,
                                       "rougeL": r.rougeL, "words": r.words, "fallback_used": r.fallback_used,
                                       "latency_seconds": None, "mean_logprob": None, "new_tokens": None,
                                       "hit_max_new_tokens": None, "chosen_from": src})
    write_jsonl(out / "selection_scores.jsonl", sel_scores)
    fields = ["pool", "method", "n_ids", "macro", "ci_lo", "ci_hi", *[f"f1_{t}" for t in TRACKS], "pool_systems", "share"]
    with (out / "selection_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    for r in sorted(summary, key=lambda r: (r["pool"], -r["macro"])):
        print(f"  {r['pool']:7} {r['method']:22} macro {r['macro']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  {r['share']}")


def row(method, pool_name, chosen_F, n_ids, pool, idx):
    F = chosen_F.reshape(n_ids, len(TRACKS))
    m, lo, hi = macro_ci(F)
    shares = {s: round(float((idx == c).mean()), 3) for c, s in enumerate(pool)}
    return {"pool": pool_name, "method": method, "n_ids": n_ids, "macro": m, "ci_lo": lo, "ci_hi": hi,
            **{f"f1_{t}": v for t, v in zip(TRACKS, per_track_means(F))}, "pool_systems": ",".join(pool),
            "share": shares if not method.startswith("single") else ""}


if __name__ == "__main__":
    main()
