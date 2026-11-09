"""Experiment B analysis: direct (raw answer from and to the language) vs translate-pivot, official F1 and LLM judge.

    python experiments/exp_b_translate_pivot/analyze.py            # after scoring and judging (see README)
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
ARABIC = ["msa", "arz", "ajp"]
TN = {"msa": "MSA", "arz": "Egyptian", "ajp": "Levantine", "en": "English"}
DIRECT, PIVOT = "sft_S2", "pivot_s2"
LAB = {DIRECT: "Direct (answer in the language)", PIVOT: "Translate-pivot (to English, answer, back)"}
COL = {DIRECT: "#1b7837", PIVOT: "#d95f02"}


def boot(x, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])


def paired(a, b, n=3000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(d.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=HERE / "results" / "analysis")
    ap.add_argument("--assembled", type=Path, default=HERE / "results" / "assembled")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "report")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = pd.read_json(a.analysis / "per_item_scores.jsonl", lines=True)
    j = pd.read_csv(a.analysis / "judge" / "judgements_flat.csv")
    first = set(pd.read_json(a.assembled / "predictions" / PIVOT / "dev_msa.jsonl", lines=True).id)
    items = items[items.id.isin(first)]
    ar = items[items.system.isin([DIRECT, PIVOT]) & items.track.isin(ARABIC)]
    jj = j[j.system.isin([DIRECT, PIVOT]) & j.track.isin(ARABIC) & j.id.isin(first)]
    f1 = ar.groupby(["system", "track"]).mm_f1.mean().unstack()
    summary = {}

    # 1. official F1 per track + overall
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    ax, w = axes[0], 0.38
    for i, s in enumerate([DIRECT, PIVOT]):
        v = [f1.loc[s, t] for t in ARABIC]
        for r, x in zip(ax.bar(np.arange(3) + (i - .5) * w, v, w, label=LAB[s], color=COL[s]), v):
            ax.text(r.get_x() + r.get_width() / 2, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3), [TN[t] for t in ARABIC]); ax.set_ylim(0.55, 0.82); ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("official BERTScore-F1"); ax.set_title("Official score per track (100 items each)")
    ax = axes[1]
    for i, s in enumerate([DIRECT, PIVOT]):
        x = ar[ar.system == s].mm_f1.values
        m = f1.loc[s].mean(); lo, hi = boot(x)
        ax.barh(i, m, color=COL[s], xerr=[[m - lo], [hi - m]], capsize=4); ax.text(hi + .002, i, f"{m:.4f}", va="center")
        summary[s] = {"macro_f1": float(m)}
    ax.set_yticks(range(2), ["Direct", "Pivot"]); ax.set_xlim(0.6, 0.8); ax.invert_yaxis(); ax.set_title("Macro over the 3 Arabic tracks")
    fig.tight_layout(); fig.savefig(a.out / "b1_official_f1.png", dpi=150); plt.close(fig)

    # 2. judge verdicts + correct per track
    v = jj.groupby("system").verdict.value_counts(normalize=True).unstack().fillna(0) * 100
    cor = jj.assign(c=jj.verdict.eq("correct") * 100).groupby(["system", "track"]).c.mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]; left = np.zeros(2)
    for vd, c in [("correct", "#4daf4a"), ("partially_correct", "#ffd92f"), ("incorrect", "#e41a1c")]:
        vals = np.array([v.loc[s, vd] if vd in v.columns else 0 for s in (DIRECT, PIVOT)])
        ax.barh(range(2), vals, left=left, color=c, label=vd.replace("_", " "))
        for k, (x0, x) in enumerate(zip(left, vals)):
            if x > 4: ax.text(x0 + x / 2, k, f"{x:.0f}%", ha="center", va="center")
        left += vals
    ax.set_yticks(range(2), ["Direct", "Pivot"]); ax.invert_yaxis(); ax.set_xlabel("% of 300 judged answers")
    ax.set_title("LLM judge verdict"); ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax = axes[1]
    for i, s in enumerate([DIRECT, PIVOT]):
        x = [cor.loc[s, t] for t in ARABIC]
        for r, y in zip(ax.bar(np.arange(3) + (i - .5) * w, x, w, label=LAB[s], color=COL[s]), x):
            ax.text(r.get_x() + r.get_width() / 2, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3), [TN[t] for t in ARABIC]); ax.set_ylim(0, 80); ax.set_ylabel("% judged correct"); ax.set_title("Correct per track")
    for s in (DIRECT, PIVOT):
        summary[s]["judge_correct_pct"] = float(v.loc[s, "correct"])
    fig.tight_layout(); fig.savefig(a.out / "b2_judge.png", dpi=150); plt.close(fig)

    # 3. paired differences (pivot - direct), per track and overall
    key = ar.set_index(["system", "track", "id"]).mm_f1
    jkey = jj.assign(c=jj.verdict.eq("correct").astype(float)).set_index(["system", "track", "id"]).c
    rows = []
    for scope in ARABIC + ["all"]:
        ts = ARABIC if scope == "all" else [scope]
        ids = [(t, i) for t in ts for i in first]
        fa = [key[(PIVOT, t, i)] - key[(DIRECT, t, i)] for t, i in ids]
        ja = [jkey[(PIVOT, t, i)] - jkey[(DIRECT, t, i)] for t, i in ids if (PIVOT, t, i) in jkey and (DIRECT, t, i) in jkey]
        d1, l1, h1 = paired(fa, np.zeros(len(fa))); d2, l2, h2 = paired(ja, np.zeros(len(ja)))
        rows.append({"scope": TN.get(scope, "All 3 tracks"), "n": len(ids), "d_F1": d1, "F1_lo": l1, "F1_hi": h1,
                     "d_judge_pts": 100 * d2, "judge_lo": 100 * l2, "judge_hi": 100 * h2})
    pd.DataFrame(rows).to_csv(a.out / "paired_pivot_minus_direct.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (c, lo, hi, ttl, fmt) in zip(axes, [("d_F1", "F1_lo", "F1_hi", "Pivot minus direct: official F1 (95% CI)", "{:+.3f}"),
                                                ("d_judge_pts", "judge_lo", "judge_hi", "Pivot minus direct: judge-correct points", "{:+.1f}")]):
        for i, r in enumerate(rows):
            ax.errorbar(r[c], i, xerr=[[r[c] - r[lo]], [r[hi] - r[c]]], fmt="o", color="k", capsize=4)
            ax.text(r[c], i - .25, fmt.format(r[c]), ha="center", fontsize=8)
        ax.axvline(0, color="grey"); ax.set_yticks(range(len(rows)), [r["scope"] for r in rows]); ax.invert_yaxis(); ax.set_title(ttl, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "b3_paired_differences.png", dpi=150); plt.close(fig)
    summary["paired"] = rows

    # 4. answer shape and dialect fidelity
    words = ar.groupby(["system", "track"]).words.mean().unstack()
    ref_words = ar[ar.system == DIRECT].assign(rw=lambda d: d.reference.str.split().str.len()).groupby("track").rw.mean()
    var = jj[jj.track != "msa"].groupby(["system", "track"]).variety_ok.mean().unstack() * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for i, s in enumerate([DIRECT, PIVOT]):
        ax.bar(np.arange(3) + (i - 1) * 0.27, [words.loc[s, t] for t in ARABIC], 0.27, color=COL[s], label=LAB[s])
    ax.bar(np.arange(3) + 0.27, [ref_words[t] for t in ARABIC], 0.27, color="#999", label="Reference")
    ax.set_xticks(range(3), [TN[t] for t in ARABIC]); ax.set_ylabel("words per answer"); ax.legend(fontsize=8); ax.set_title("Answer length")
    ax = axes[1]
    for i, s in enumerate([DIRECT, PIVOT]):
        x = [var.loc[s, t] for t in ["arz", "ajp"]]
        for r, y in zip(ax.bar(np.arange(2) + (i - .5) * w, x, w, color=COL[s], label=LAB[s]), x):
            ax.text(r.get_x() + r.get_width() / 2, y, f"{y:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(2), ["Egyptian", "Levantine"]); ax.set_ylim(0, 110); ax.set_ylabel("% answers in the requested dialect (judge)")
    ax.set_title("Dialect fidelity (is the answer really Egyptian / Levantine?)")
    fig.tight_layout(); fig.savefig(a.out / "b4_shape_and_dialect.png", dpi=150); plt.close(fig)

    # 5. where does the pivot lose? English-answer quality vs final
    en = items[items.system.str.startswith("pivot_en_")].assign(src=lambda d: d.system.str.removeprefix("pivot_en_"))
    en_f1 = en.groupby("src").mm_f1.mean()
    direct_en = items[(items.system == DIRECT) & (items.track == "en")].mm_f1.mean()
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(3); w3 = 0.27
    stages = [("Direct (native answer)", [f1.loc[DIRECT, t] for t in ARABIC], COL[DIRECT]),
              ("Pivot: English answer vs English reference", [en_f1[t] for t in ARABIC], "#7570b3"),
              ("Pivot: final answer after translating back", [f1.loc[PIVOT, t] for t in ARABIC], COL[PIVOT])]
    for i, (lab, vals, c) in enumerate(stages):
        for r, y in zip(ax.bar(x + (i - 1) * w3, vals, w3, label=lab, color=c), vals):
            ax.text(r.get_x() + r.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(direct_en, color="#1b7837", ls="--", lw=1); ax.text(2.45, direct_en + .002, f"direct English track: {direct_en:.3f}", ha="right", fontsize=8, color="#1b7837")
    ax.set_xticks(x, [f"questions from {TN[t]}" for t in ARABIC]); ax.set_ylim(0.6, 0.9); ax.set_ylabel("official BERTScore-F1")
    ax.set_title("Where does the pivot lose? (different references per bar, see README)"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "b5_pipeline_stages.png", dpi=150); plt.close(fig)
    summary["english_stage_f1"] = {t: float(en_f1[t]) for t in ARABIC}; summary["direct_english_f1"] = float(direct_en)

    # 6. per-item scatter
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for ax, t in zip(axes, ARABIC):
        d = np.array([key[(DIRECT, t, i)] for i in sorted(first)]); p = np.array([key[(PIVOT, t, i)] for i in sorted(first)])
        ax.scatter(d, p, s=14, alpha=.6, color="#444"); ax.plot([.5, 1], [.5, 1], "k--", lw=1)
        ax.set_title(f"{TN[t]}: pivot wins {np.mean(p > d) * 100:.0f}% / direct wins {np.mean(d > p) * 100:.0f}%", fontsize=10)
        ax.set_xlabel("direct F1")
    axes[0].set_ylabel("pivot F1"); fig.tight_layout(); fig.savefig(a.out / "b6_per_item_scatter.png", dpi=150); plt.close(fig)

    # 7. judge rubric
    dims = ["helpfulness", "relevance", "accuracy", "faithfulness"]
    dm = jj.groupby("system")[dims].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate([DIRECT, PIVOT]):
        ax.bar(np.arange(4) + (i - .5) * w, dm.loc[s, dims], w, color=COL[s], label=LAB[s])
    ax.set_xticks(range(4), dims); ax.set_ylim(0, 9); ax.set_ylabel("mean judge score (0-10)"); ax.legend(fontsize=8); ax.set_title("LLM judge rubric (3 Arabic tracks)")
    fig.tight_layout(); fig.savefig(a.out / "b7_judge_dimensions.png", dpi=150); plt.close(fig)

    # examples
    asm = {t: pd.read_json(a.assembled / "predictions" / PIVOT / f"dev_{t}.jsonl", lines=True).set_index("id") for t in ARABIC}
    ex = []
    for t in ARABIC:
        for i in sorted(first)[:8]:
            r = asm[t].loc[i]; d = ar[(ar.system == DIRECT) & (ar.track == t) & (ar.id == i)].iloc[0]
            ex.append({"track": t, "question": r.question, "question_en": r.question_en, "answer_en": r.answer_en,
                       "pivot_answer": r.answer, "direct_answer": d.prediction, "reference": d.reference,
                       "pivot_f1": key[(PIVOT, t, i)], "direct_f1": key[(DIRECT, t, i)]})
    pd.DataFrame(ex).to_csv(a.out / "examples.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(json.dumps(summary, indent=1, default=float)[:1500])


if __name__ == "__main__":
    main()
