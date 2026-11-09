"""Extra comparison charts: fine-tuned Qwen vs base Qwen (both runs) vs CulturalPangea, official F1 and LLM judge.

    python -m local_analysis.charts_extra --analysis results/analysis/r1 --out results/report/charts_extra
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYSTEMS = [("sft_S2", "Qwen fine-tuned S2", "#1b7837"), ("sft_S1", "Qwen fine-tuned S1", "#7fbf7b"),
           ("qwen_z1", "Qwen base (new run)", "#2166ac"), ("hist_qwen3_5_4b", "Qwen base (old run)", "#92c5de"),
           ("pangea_p0", "CulturalPangea (rerun)", "#b2182b"), ("hist_culturalpangea_7b", "CulturalPangea (old run)", "#ef8a62")]
TRACKS = ["en", "msa", "arz", "ajp"]
TNAME = {"en": "English", "msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def grouped(ax, table, ylabel, title, fmt="{:.2f}"):
    w = 0.8 / len(SYSTEMS)
    for i, (k, lab, col) in enumerate(SYSTEMS):
        xs = np.arange(len(TRACKS)) + i * w - 0.4 + w / 2
        vals = [table.loc[k, t] for t in TRACKS]
        b = ax.bar(xs, vals, w, label=lab, color=col)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(TRACKS)), [TNAME[t] for t in TRACKS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(a.analysis / "judge" / "judgements_flat.csv")
    j = j[j.system.isin([s[0] for s in SYSTEMS])]
    sc = pd.read_json(a.analysis / "per_item_scores.jsonl", lines=True)
    key = {k: lab for k, lab, _ in SYSTEMS}
    col = {k: c for k, _, c in SYSTEMS}

    # 1 official F1 per track, same 100 items/track the judge saw (old runs only have these)
    f1 = j.groupby(["system", "track"]).mm_f1.mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped(ax, f1, "official BERTScore-F1", "Official score per track (100 items/track, same items for all six runs)", "{:.3f}")
    ax.set_ylim(0.6, 0.9)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "x1_official_f1_by_track.png", dpi=150); plt.close(fig)

    # 2 macro F1 with CI: left = common 400 items, right = dev500 (new runs only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [6, 4]})
    ax = axes[0]
    for i, (k, lab, c) in enumerate(SYSTEMS):
        s = j[j.system == k]
        m = s.groupby("track").mm_f1.mean().mean()
        # per-id mean over tracks then bootstrap items
        lo, hi = boot_ci(s.mm_f1.values)
        ax.barh(i, m, color=c, xerr=[[m - lo], [hi - m]], capsize=3)
        ax.text(hi + 0.002, i, f"{m:.4f}", va="center", fontsize=8)
    ax.set_yticks(range(len(SYSTEMS)), [s[1] for s in SYSTEMS]); ax.invert_yaxis()
    ax.set_xlim(0.66, 0.82); ax.set_xlabel("macro F1 (95% bootstrap CI)")
    ax.set_title("Macro F1, 400 shared items")
    ax = axes[1]
    new = [s for s in SYSTEMS if not s[0].startswith("hist_")]
    for i, (k, lab, c) in enumerate(new):
        s = sc[sc.system == k]
        m = s.groupby("track").mm_f1.mean().mean()
        lo, hi = boot_ci(s.mm_f1.values)
        ax.barh(i, m, color=c, xerr=[[m - lo], [hi - m]], capsize=3)
        ax.text(hi + 0.002, i, f"{m:.4f}", va="center", fontsize=8)
    ax.set_yticks(range(len(new)), [s[1] for s in new]); ax.invert_yaxis()
    ax.set_xlim(0.66, 0.82); ax.set_xlabel("macro F1"); ax.set_title("Macro F1, dev500 (new runs)")
    fig.tight_layout(); fig.savefig(a.out / "x2_official_macro_ci.png", dpi=150); plt.close(fig)

    # 3 judge verdict stacked
    v = j.groupby("system").verdict.value_counts(normalize=True).unstack().fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    left = np.zeros(len(SYSTEMS))
    for verdict, c in [("correct", "#4daf4a"), ("partially_correct", "#ffd92f"), ("incorrect", "#e41a1c")]:
        vals = np.array([v.loc[k, verdict] if verdict in v.columns else 0 for k, _, _ in SYSTEMS])
        ax.barh(range(len(SYSTEMS)), vals, left=left, color=c, label=verdict.replace("_"," "))
        for i, (x0, x) in enumerate(zip(left, vals)):
            if x > 4: ax.text(x0 + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=9)
        left += vals
    ax.set_yticks(range(len(SYSTEMS)), [s[1] for s in SYSTEMS]); ax.invert_yaxis()
    ax.set_xlabel("% of 400 judged answers"); ax.set_title("LLM judge verdict (gemini-2.5-flash)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(); fig.savefig(a.out / "x3_judge_verdicts.png", dpi=150); plt.close(fig)

    # 4 judge correct% per track
    cor = (j.assign(c=j.verdict.eq("correct") * 100).groupby(["system", "track"]).c.mean().unstack())
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped(ax, cor, "% judged correct", "LLM judge: share of answers judged correct, per track (100/track)", "{:.0f}")
    ax.set_ylim(0, 85); ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "x4_judge_correct_by_track.png", dpi=150); plt.close(fig)

    # 5 judge score dimensions
    dims = ["helpfulness", "relevance", "accuracy", "faithfulness"]
    d = j.groupby("system")[dims].mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.8 / len(SYSTEMS)
    for i, (k, lab, c) in enumerate(SYSTEMS):
        ax.bar(np.arange(4) + i * w - 0.4 + w / 2, d.loc[k, dims], w, label=lab, color=c)
    ax.set_xticks(range(4), dims); ax.set_ylabel("mean judge score (0-10)"); ax.set_ylim(0, 9)
    ax.set_title("LLM judge rubric scores"); ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(a.out / "x5_judge_dimensions.png", dpi=150); plt.close(fig)

    # 6 metric vs judge: where the two disagree
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k, lab, c in SYSTEMS:
        s = j[j.system == k]
        ax.scatter(s.mm_f1.mean(), s.verdict.eq("correct").mean() * 100, s=140, color=c, edgecolor="k", label=lab)
    ax.set_xlabel("official BERTScore-F1 (400 shared items)"); ax.set_ylabel("% judged correct")
    ax.set_title("Official score vs LLM judge"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.out / "x6_metric_vs_judge.png", dpi=150); plt.close(fig)

    # 7 official F1 spread by verdict (does the metric track correctness?)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data, labels = [], []
    for verdict in ["incorrect", "partially_correct", "correct"]:
        data.append(j[j.verdict == verdict].mm_f1.values); labels.append(f"{verdict}\n(n={len(data[-1])})")
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("official BERTScore-F1"); ax.set_title("Official F1 by judge verdict (all six systems pooled)")
    fig.tight_layout(); fig.savefig(a.out / "x7_f1_by_verdict.png", dpi=150); plt.close(fig)
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
