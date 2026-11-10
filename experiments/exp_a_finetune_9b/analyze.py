"""Experiment A analysis: 4B base, 9B base, 4B fine-tuned, 9B fine-tuned (official F1, LLM judge, training, cost).

    python experiments/exp_a_finetune_9b/analyze.py        # after scoring and judging (see README)
Charts and tables go to results/report/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRACKS = ["en", "msa", "arz", "ajp"]
TN = {"en": "English", "msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}
SYS = [("sft9b_S2", "9B fine-tuned (S2, lr 2e-4)", "#8c510a"), ("sft9b_S1", "9B fine-tuned (S1, lr 1e-4)", "#d8b365"),
       ("sft_S2", "4B fine-tuned (S2, lr 2e-4)", "#1b7837"), ("sft_S1", "4B fine-tuned (S1, lr 1e-4)", "#7fbf7b"),
       ("qwen9b_z1", "9B base", "#e08214"), ("qwen_z1", "4B base", "#2166ac")]
NAME = {k: l for k, l, _ in SYS}
COLOR = {k: c for k, _, c in SYS}


def boot(x, n=3000, seed=0):
    x = np.asarray(x, float)
    return np.percentile(np.random.default_rng(seed).choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def paired(a, b, n=3000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(d.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


def grouped(ax, table, keys, ylabel, title, fmt):
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        v = [table.loc[k, t] for t in TRACKS]
        for r, y in zip(ax.bar(np.arange(4) + i * w - .4 + w / 2, v, w, label=NAME[k], color=COLOR[k]), v):
            ax.text(r.get_x() + r.get_width() / 2, y, fmt.format(y), ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(range(4), [TN[t] for t in TRACKS]); ax.set_ylabel(ylabel); ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=HERE / "results" / "analysis")
    ap.add_argument("--kaggle", type=Path, default=HERE / "results" / "kaggle")
    ap.add_argument("--phase1", type=Path, default=REPO / "results" / "phase1" / "full")
    ap.add_argument("--phase2", type=Path, default=REPO / "results" / "phase2" / "full")
    ap.add_argument("--phase3", type=Path, default=REPO / "results" / "phase3" / "full")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "report")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    j = pd.read_csv(a.analysis / "judge" / "judgements_flat.csv")
    j = j[j.system.isin(NAME)]
    keys = [k for k, _, _ in SYS if k in set(j.system)]
    f1 = j.groupby(["system", "track"]).mm_f1.mean().unstack()
    verdict = j.groupby("system").verdict.value_counts(normalize=True).unstack().fillna(0) * 100
    cor = j.assign(c=j.verdict.eq("correct") * 100).groupby(["system", "track"]).c.mean().unstack()
    summary = {k: {"macro_f1": float(f1.loc[k].mean()), "judge_correct_pct": float(verdict.loc[k, "correct"])} for k in keys}

    # 1 official F1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    grouped(axes[0], f1, keys, "official BERTScore-F1", "Official score per track (100 items/track, identical items)", "{:.3f}")
    axes[0].set_ylim(0.6, 0.9); axes[0].legend(fontsize=7, loc="upper right")
    for i, k in enumerate(keys):
        x = j[j.system == k].mm_f1.values; m = f1.loc[k].mean(); lo, hi = boot(x)
        axes[1].barh(i, m, color=COLOR[k], xerr=[[m - lo], [hi - m]], capsize=3); axes[1].text(hi + .002, i, f"{m:.4f}", va="center", fontsize=8)
    axes[1].set_yticks(range(len(keys)), [NAME[k] for k in keys], fontsize=8); axes[1].invert_yaxis(); axes[1].set_xlim(0.66, 0.84)
    axes[1].set_xlabel("macro F1 (95% CI)"); axes[1].set_title("Macro F1")
    fig.tight_layout(); fig.savefig(a.out / "a1_official_f1.png", dpi=150); plt.close(fig)

    # 2 judge
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 3]})
    ax, left = axes[0], np.zeros(len(keys))
    for vd, c in [("correct", "#4daf4a"), ("partially_correct", "#ffd92f"), ("incorrect", "#e41a1c")]:
        vals = np.array([verdict.loc[k, vd] if vd in verdict.columns else 0 for k in keys])
        ax.barh(range(len(keys)), vals, left=left, color=c, label=vd.replace("_", " "))
        for i, (x0, x) in enumerate(zip(left, vals)):
            if x > 4: ax.text(x0 + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=9)
        left += vals
    ax.set_yticks(range(len(keys)), [NAME[k] for k in keys], fontsize=8); ax.invert_yaxis(); ax.set_xlabel("% of 400 judged answers")
    ax.set_title("LLM judge verdict"); ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    grouped(axes[1], cor, keys, "% judged correct", "LLM judge: correct per track", "{:.0f}"); axes[1].set_ylim(0, 90); axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "a2_judge.png", dpi=150); plt.close(fig)

    # 3 size x fine-tuning interaction
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, vals, ttl in [(axes[0], {k: summary[k]["macro_f1"] for k in keys}, "Official macro F1"),
                          (axes[1], {k: summary[k]["judge_correct_pct"] for k in keys}, "Judge: % correct")]:
        for size, base, ft, c in [("4B", "qwen_z1", "sft_S2", "#2166ac"), ("9B", "qwen9b_z1", "sft9b_S2", "#e08214")]:
            if base in vals and ft in vals:
                ax.plot(["base", "fine-tuned"], [vals[base], vals[ft]], "o-", color=c, lw=2, label=f"Qwen3.5-{size} (NF4)")
                for xi, y in enumerate([vals[base], vals[ft]]):
                    ax.annotate(f"{y:.3f}" if "F1" in ttl else f"{y:.1f}", (xi, y), textcoords="offset points",
                                xytext=(-26 if size == "4B" else 26, 6 if size == "4B" else -12), ha="center", fontsize=8, color=c)
        ax.set_title(ttl); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Does model size still matter after fine-tuning?"); fig.tight_layout(); fig.savefig(a.out / "a3_size_vs_finetune.png", dpi=150); plt.close(fig)

    # 4 paired differences
    key = j.set_index(["system", "track", "id"])
    pairs = [("sft9b_S2", "sft_S2", "9B fine-tuned minus 4B fine-tuned"), ("sft9b_S2", "qwen9b_z1", "9B fine-tuned minus 9B base"),
             ("sft_S2", "qwen_z1", "4B fine-tuned minus 4B base"), ("qwen9b_z1", "qwen_z1", "9B base minus 4B base")]
    rows = []
    for x, y, lab in pairs:
        if x in keys and y in keys:
            ix = key.loc[x].index.intersection(key.loc[y].index)
            fx, fy = key.loc[x].loc[ix].mm_f1.values, key.loc[y].loc[ix].mm_f1.values
            cx, cy = [(key.loc[s].loc[ix].verdict == "correct").astype(float).values for s in (x, y)]
            d1, l1, h1 = paired(fx, fy); d2, l2, h2 = paired(cx, cy)
            rows.append({"comparison": lab, "n": len(ix), "d_macro_F1": d1, "F1_lo": l1, "F1_hi": h1,
                         "d_judge_pts": 100 * d2, "judge_lo": 100 * l2, "judge_hi": 100 * h2})
    pd.DataFrame(rows).to_csv(a.out / "paired_differences.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (c, lo, hi, ttl, fmt) in zip(axes, [("d_macro_F1", "F1_lo", "F1_hi", "Official F1 difference (paired, 95% CI)", "{:+.3f}"),
                                                ("d_judge_pts", "judge_lo", "judge_hi", "Judge-correct difference, points (95% CI)", "{:+.1f}")]):
        for i, r in enumerate(rows):
            ax.errorbar(r[c], i, xerr=[[r[c] - r[lo]], [r[hi] - r[c]]], fmt="o", color="k", capsize=4); ax.text(r[c], i + .3, fmt.format(r[c]), ha="center", fontsize=8)
        ax.axvline(0, color="grey"); ax.set_yticks(range(len(rows)), [r["comparison"] for r in rows], fontsize=8); ax.invert_yaxis(); ax.set_title(ttl, fontsize=10, pad=10)
    fig.tight_layout(); fig.savefig(a.out / "a4_paired_differences.png", dpi=150); plt.close(fig)
    summary["paired"] = rows

    # 5 training curves and cost
    logs4, logs9 = a.phase1 / "results" / "logs", (a.kaggle / "logs" if (a.kaggle / "logs").exists() else a.kaggle / "results" / "logs")
    train = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, d, c, ls in [("4B S2", logs4 / "sft_S2", "#1b7837", "-"), ("4B S1", logs4 / "sft_S1", "#7fbf7b", "--"),
                           ("9B S2", logs9 / "sft_S2_9B", "#8c510a", "-"), ("9B S1", logs9 / "sft_S1_9B", "#d8b365", "--")]:
        h, s = Path(f"{d}_history.csv"), Path(f"{d}_summary.json")
        if not h.exists():
            continue
        hist = pd.read_csv(h)
        axes[0].plot(hist.step, hist.loss.rolling(8, min_periods=1).mean(), color=c, ls=ls, label=name)
        ev = hist.dropna(subset=["eval_loss"]); axes[1].plot(ev.step, ev.eval_loss, "o-", color=c, ls=ls, label=name)
        sm = json.loads(s.read_text()); train[name] = {k: sm.get(k) for k in ("steps_done", "best_eval_loss", "hours", "sec_per_step", "peak_mem_gb", "fraction_of_epoch")}
    axes[0].set_title("Training loss (smoothed)"); axes[1].set_title("Held-out eval loss"); axes[0].set_xlabel("optimizer step"); axes[1].set_xlabel("optimizer step")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    if train:
        names = list(train); vals = [train[n]["sec_per_step"] for n in names]
        axes[2].bar(names, vals, color=["#1b7837", "#7fbf7b", "#8c510a", "#d8b365"][:len(names)])
        for xi, y in enumerate(vals): axes[2].text(xi, y, f"{y:.0f}s", ha="center", va="bottom")
        axes[2].set_title("Seconds per optimizer step (batch 16)")
    fig.tight_layout(); fig.savefig(a.out / "a5_training.png", dpi=150); plt.close(fig)
    summary["training"] = train
    pd.DataFrame(train).T.to_csv(a.out / "training.csv")

    # 6 rubric + dialect fidelity
    dims = ["helpfulness", "relevance", "accuracy", "faithfulness"]
    dm = j.groupby("system")[dims].mean(); var = j[j.track != "en"].groupby("system").variety_ok.mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 2]}); w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        axes[0].bar(np.arange(4) + i * w - .4 + w / 2, dm.loc[k, dims], w, color=COLOR[k], label=NAME[k])
    axes[0].set_xticks(range(4), dims); axes[0].set_ylim(0, 9); axes[0].set_title("LLM judge rubric"); axes[0].legend(fontsize=7, ncol=2)
    axes[1].barh(range(len(keys)), [var[k] for k in keys], color=[COLOR[k] for k in keys])
    for i, k in enumerate(keys): axes[1].text(var[k] + 1, i, f"{var[k]:.0f}%", va="center", fontsize=8)
    axes[1].set_yticks(range(len(keys)), [NAME[k] for k in keys], fontsize=8); axes[1].invert_yaxis(); axes[1].set_xlim(0, 112)
    axes[1].set_title("Arabic answers in the requested variety (judge)")
    fig.tight_layout(); fig.savefig(a.out / "a6_rubric_and_dialect.png", dpi=150); plt.close(fig)

    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(json.dumps({k: summary[k] for k in keys}, indent=1)); print(pd.DataFrame(train).T)


if __name__ == "__main__":
    main()
