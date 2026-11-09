"""Qwen3.5-4B base vs Qwen3.5-9B (NF4) base vs fine-tuned 4B: official F1, LLM judge, speed and memory.

    python -m local_analysis.compare_9b --analysis results/analysis/r1 --p2 results/phase2/full --p3 results/phase3/full \
        --out results/report/qwen9b
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYS = [("sft_S2", "Qwen3.5-4B fine-tuned (S2)", "#1b7837"), ("sft_S1", "Qwen3.5-4B fine-tuned (S1)", "#7fbf7b"),
       ("qwen_z1", "Qwen3.5-4B base (NF4)", "#2166ac"), ("qwen9b_z1", "Qwen3.5-9B base (NF4)", "#e08214"),
       ("hist_qwen3_5_9b", "Qwen3.5-9B base, old run (old prompt)", "#fdb863")]
TRACKS = ["en", "msa", "arz", "ajp"]
TN = {"en": "English", "msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}
KEY = [s[0] for s in SYS]


def boot(x, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def paired(a, b, n=3000, seed=1):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bars(ax, table, ylabel, title, fmt):
    w = 0.8 / len(SYS)
    for i, (k, lab, c) in enumerate(SYS):
        if k not in table.index:
            continue
        xs = np.arange(4) + i * w - 0.4 + w / 2
        v = [table.loc[k, t] for t in TRACKS]
        for r, x in zip(ax.bar(xs, v, w, label=lab, color=c), v):
            ax.text(r.get_x() + r.get_width() / 2, x, fmt.format(x), ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(range(4), [TN[t] for t in TRACKS]); ax.set_ylabel(ylabel); ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--p2", type=Path, required=True)
    ap.add_argument("--p3", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(a.analysis / "judge" / "judgements_flat.csv")
    j = j[j.system.isin(KEY)]
    j["sys_id"] = j.system
    # judge flat csv has only these columns; official F1 for the same items comes from mm_f1 in it
    f1 = j.groupby(["system", "track"]).mm_f1.mean().unstack()
    present = [k for k in KEY if k in f1.index]
    out = {}

    # 1 official F1 per track + macro with CI
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    bars(axes[0], f1, "official BERTScore-F1", "Official score per track (100 items/track, identical items)", "{:.3f}")
    axes[0].set_ylim(0.6, 0.9); axes[0].legend(fontsize=7, loc="upper right")
    ax = axes[1]
    for i, k in enumerate(present):
        s = j[j.system == k]
        m = s.groupby("track").mm_f1.mean().mean(); lo, hi = boot(s.mm_f1.values)
        ax.barh(i, m, color=dict((x[0], x[2]) for x in SYS)[k], xerr=[[m - lo], [hi - m]], capsize=3)
        ax.text(hi + 0.002, i, f"{m:.4f}", va="center", fontsize=8)
        out.setdefault(k, {})["macro_f1"] = m
    ax.set_yticks(range(len(present)), [dict((x[0], x[1]) for x in SYS)[k] for k in present], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0.66, 0.84); ax.set_xlabel("macro F1 (95% CI)"); ax.set_title("Macro F1")
    fig.tight_layout(); fig.savefig(a.out / "q1_official_f1.png", dpi=150); plt.close(fig)

    # 2 judge verdicts and per-track correct
    v = j.groupby("system").verdict.value_counts(normalize=True).unstack().fillna(0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 3]})
    ax = axes[0]; left = np.zeros(len(present))
    for verdict, c in [("correct", "#4daf4a"), ("partially_correct", "#ffd92f"), ("incorrect", "#e41a1c")]:
        vals = np.array([v.loc[k, verdict] if verdict in v.columns else 0 for k in present])
        ax.barh(range(len(present)), vals, left=left, color=c, label=verdict.replace("_", " "))
        for i, (x0, x) in enumerate(zip(left, vals)):
            if x > 4: ax.text(x0 + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=9)
        left += vals
    ax.set_yticks(range(len(present)), [dict((x[0], x[1]) for x in SYS)[k] for k in present], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("% of 400 judged answers"); ax.set_title("LLM judge verdict"); ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    cor = j.assign(c=j.verdict.eq("correct") * 100).groupby(["system", "track"]).c.mean().unstack()
    bars(axes[1], cor, "% judged correct", "LLM judge: correct per track", "{:.0f}")
    axes[1].set_ylim(0, 85); axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "q2_judge.png", dpi=150); plt.close(fig)
    for k in present:
        out[k]["judge_correct_pct"] = float(v.loc[k, "correct"])

    # 3 rubric dimensions
    dims = ["helpfulness", "relevance", "accuracy", "faithfulness"]
    d = j.groupby("system")[dims].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.8 / len(present)
    for i, k in enumerate(present):
        ax.bar(np.arange(4) + i * w - 0.4 + w / 2, d.loc[k, dims], w, color=dict((x[0], x[2]) for x in SYS)[k],
               label=dict((x[0], x[1]) for x in SYS)[k])
    ax.set_xticks(range(4), dims); ax.set_ylim(0, 9); ax.set_ylabel("mean judge score (0-10)"); ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("LLM judge rubric scores"); fig.tight_layout(); fig.savefig(a.out / "q3_judge_dimensions.png", dpi=150); plt.close(fig)

    # 4 paired differences (same items): 9B vs 4B base, S2 vs 9B
    pairs = [("qwen9b_z1", "qwen_z1", "9B base minus 4B base"), ("sft_S2", "qwen9b_z1", "4B fine-tuned (S2) minus 9B base"),
             ("sft_S2", "qwen_z1", "4B fine-tuned (S2) minus 4B base")]
    rows = []
    key = j.set_index(["system", "track", "id"])
    for x, y, lab in pairs:
        if x in present and y in present:
            ix = key.loc[x].index.intersection(key.loc[y].index)
            fx, fy = key.loc[x].loc[ix].mm_f1.values, key.loc[y].loc[ix].mm_f1.values
            cx = (key.loc[x].loc[ix].verdict == "correct").astype(float).values
            cy = (key.loc[y].loc[ix].verdict == "correct").astype(float).values
            d1, lo1, hi1 = paired(fx, fy); d2, lo2, hi2 = paired(cx, cy)
            rows.append({"comparison": lab, "n": len(ix), "d_macro_F1": d1, "F1_lo": lo1, "F1_hi": hi1,
                         "d_judge_correct_pts": 100 * d2, "judge_lo": 100 * lo2, "judge_hi": 100 * hi2})
    pd.DataFrame(rows).to_csv(a.out / "paired_differences.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (col, lo, hi, ttl, sc) in zip(axes, [("d_macro_F1", "F1_lo", "F1_hi", "Official F1 difference (paired, 95% CI)", 1),
                                                  ("d_judge_correct_pts", "judge_lo", "judge_hi", "Judge-correct difference, points (95% CI)", 1)]):
        for i, r in enumerate(rows):
            ax.errorbar(r[col], i, xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt="o", color="k", capsize=4)
            ax.text(r[col], i - 0.25, f"{r[col]:+.3f}" if col == "d_macro_F1" else f"{r[col]:+.1f}", ha="center", fontsize=8)
        ax.axvline(0, color="grey", lw=1); ax.set_yticks(range(len(rows)), [r["comparison"] for r in rows], fontsize=8)
        ax.invert_yaxis(); ax.set_title(ttl, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "q4_paired_differences.png", dpi=150); plt.close(fig)

    # 5 cost: GPU-seconds per item and peak memory
    cost = []
    for stage, src, wall_key in [("qwen_z1", a.p2, None), ("qwen9b_z1", a.p3, None)]:
        raw = [json.loads(l) for f in (src / "results" / "raw" / stage).glob("worker_*.jsonl") for l in open(f)]
        comp = json.load(open(src / "results" / "logs" / ("phase2_completion.json" if stage == "qwen_z1" else "phase3_completion.json")))
        if stage == "qwen_z1":
            wall = comp["stages"][stage]["run"]["wall_seconds"]; n = comp["stages"][stage]["run"]["answered"]; gpus = 2
        else:
            wall = comp["run"]["wall_seconds"]; n = comp["run"]["answered"]; gpus = comp["gpus"]
        cost.append({"system": stage, "items": n, "gpu_seconds_per_item": wall * gpus / n,
                     "peak_mem_gb_per_worker": max(r["peak_mem_gb"] for r in raw if r.get("peak_mem_gb") is not None),
                     "mean_new_tokens": np.mean([r["new_tokens"] for r in raw if r.get("new_tokens")])})
    pd.DataFrame(cost).to_csv(a.out / "cost.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labs = ["4B base (NF4)", "9B base (NF4)"]; cols = ["#2166ac", "#e08214"]
    for ax, col, ttl in zip(axes, ["gpu_seconds_per_item", "peak_mem_gb_per_worker"], ["GPU-seconds per item (lower is cheaper)", "Peak GPU memory per worker, GB"]):
        vals = [c[col] for c in cost]
        for r, x in zip(ax.bar(labs, vals, color=cols), vals): ax.text(r.get_x() + r.get_width() / 2, x, f"{x:.1f}", ha="center", va="bottom")
        ax.set_title(ttl, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "q5_cost.png", dpi=150); plt.close(fig)

    (a.out / "summary.json").write_text(json.dumps({"per_system": out, "paired": rows, "cost": cost}, indent=1, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(cost).round(2).to_string(index=False))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
