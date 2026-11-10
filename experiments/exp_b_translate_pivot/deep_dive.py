"""Experiment B deep dive: why the pivot loses official F1, whether the judge gain is real, and what fixes it.

Needs diagnostics.py, then scoring and judging of its systems into results/diagnostics/analysis (see README).
Writes charts and tables to results/report/deep_dive/.

    python experiments/exp_b_translate_pivot/deep_dive.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
DIAG = HERE / "results" / "diagnostics" / "analysis"
BASE = HERE / "results" / "analysis"
OUT = HERE / "results" / "report" / "deep_dive"
ARABIC = ["msa", "arz", "ajp"]
TN = {"msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}
SYS = {  # system -> (label, colour)
    "sft_S2": ("Direct (answer in Arabic)", "#2166ac"),
    "pivot_s2": ("Pivot, plain translator", "#e08214"),
    "entrack_tr": ("Pivot, English-track question", "#fdb863"),
    "pivot_fs": ("Pivot, style-matched translator", "#1b7837"),
    "ref_tr": ("Ceiling: gold English, plain translator", "#bababa"),
    "ref_tr_fs": ("Ceiling: gold English, style-matched", "#878787"),
}
B = 2000


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def boot(x, y, rng):
    d = x - y
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    s = pd.DataFrame(read_jsonl(DIAG / "per_item_scores.jsonl"))
    keys = set(zip(*[s[s.system == "pivot_s2"][c] for c in ("track", "id")]))
    s = s[[k in keys for k in zip(s.track, s.id)]]
    j = pd.read_csv(DIAG / "judge" / "judgements_flat.csv")
    j = j[[k in keys for k in zip(j.track, j.id)]]
    jb = pd.read_csv(BASE / "judge" / "judgements_flat.csv")
    j_en = jb[jb.system.str.startswith("pivot_en_")].copy()
    j_en["track"] = j_en.system.str.replace("pivot_en_", "")

    systems = [k for k in SYS if k in set(s.system)]
    f1 = s.pivot_table(index="system", columns="track", values="mm_f1", aggfunc="mean").loc[systems, ARABIC]
    f1["macro"] = f1.mean(axis=1)
    corr = (j.assign(c=j.verdict == "correct").pivot_table(index="system", columns="track", values="c", aggfunc="mean") * 100)
    corr = corr.reindex(systems)[ARABIC]
    corr["all"] = corr.mean(axis=1)
    shape = s.groupby("system")[["mm_p", "mm_r", "rougeL", "words"]].mean().loc[systems]
    table = pd.concat({"F1": f1, "judge_correct_pct": corr}, axis=1).round(4)
    table.to_csv(OUT / "systems.csv")

    # paired differences against direct
    rows = []
    for sysname in ["pivot_s2", "entrack_tr", "pivot_fs"]:
        for scope in ARABIC + ["all"]:
            tr = ARABIC if scope == "all" else [scope]
            a = s[(s.system == sysname) & s.track.isin(tr)].set_index(["track", "id"]).mm_f1
            b = s[(s.system == "sft_S2") & s.track.isin(tr)].set_index(["track", "id"]).mm_f1.loc[a.index]
            ja = j[(j.system == sysname) & j.track.isin(tr)].set_index(["track", "id"]).verdict.eq("correct").astype(float)
            jbb = j[(j.system == "sft_S2") & j.track.isin(tr)].set_index(["track", "id"]).verdict.eq("correct").astype(float)
            ix = ja.index.intersection(jbb.index)
            d1 = boot(a.values, b.values, rng)
            d2 = boot(ja.loc[ix].values, jbb.loc[ix].values, rng)
            rows.append({"system": sysname, "scope": TN.get(scope, "All 3 tracks"), "d_F1": d1[0], "F1_lo": d1[1], "F1_hi": d1[2],
                         "d_judge_pts": 100 * d2[0], "judge_lo": 100 * d2[1], "judge_hi": 100 * d2[2]})
    paired = pd.DataFrame(rows)
    paired.round(4).to_csv(OUT / "paired_vs_direct.csv", index=False)

    # transitions direct -> pivot (plain) and direct -> pivot_fs
    trans = {}
    for sysname in ["pivot_s2", "pivot_fs"]:
        a = j[j.system == "sft_S2"].set_index(["track", "id"]).verdict
        b = j[j.system == sysname].set_index(["track", "id"]).verdict
        ix = a.index.intersection(b.index)
        trans[sysname] = pd.crosstab(a.loc[ix].rename("direct"), b.loc[ix].rename(sysname)).reindex(
            index=["correct", "partially_correct", "incorrect"], columns=["correct", "partially_correct", "incorrect"], fill_value=0)

    # both-correct items: F1 gap when content is equally right
    jd = j[j.system == "sft_S2"].set_index(["track", "id"]).verdict
    jp = j[j.system == "pivot_s2"].set_index(["track", "id"]).verdict
    ix = jd.index.intersection(jp.index)
    both = ix[((jd.loc[ix] == "correct") & (jp.loc[ix] == "correct")).values]
    sd = s[s.system == "sft_S2"].set_index(["track", "id"])
    sp = s[s.system == "pivot_s2"].set_index(["track", "id"])
    sf = s[s.system == "pivot_fs"].set_index(["track", "id"])
    both_tbl = pd.DataFrame({"direct": sd.loc[both, ["mm_f1", "mm_p", "mm_r", "rougeL"]].mean(),
                             "pivot_plain": sp.loc[both, ["mm_f1", "mm_p", "mm_r", "rougeL"]].mean(),
                             "pivot_style": sf.loc[both, ["mm_f1", "mm_p", "mm_r", "rougeL"]].mean()}).round(4)
    both_tbl.to_csv(OUT / "both_correct_items.csv")

    en_stage = (j_en.assign(c=j_en.verdict == "correct").groupby("track").c.mean() * 100).reindex(ARABIC)

    # ---------- charts ----------
    # d1: both metrics, all systems
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    lab = [SYS[k][0] for k in systems]
    col = [SYS[k][1] for k in systems]
    y = np.arange(len(systems))
    axes[0].barh(y, f1["macro"], color=col)
    for i, v in enumerate(f1["macro"]):
        axes[0].text(v + .002, i, f"{v:.4f}", va="center", fontsize=9)
    axes[0].axvline(f1.loc["sft_S2", "macro"], color="#2166ac", ls=":", lw=1)
    axes[0].set_yticks(y, lab); axes[0].invert_yaxis(); axes[0].set_xlim(.74, .91)
    axes[0].set_title("Official macro F1 (3 Arabic tracks, 300 items)"); axes[0].set_xlabel("BERTScore-F1")
    axes[1].barh(y, corr["all"], color=col)
    for i, v in enumerate(corr["all"]):
        axes[1].text(v + .8, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[1].axvline(corr.loc["sft_S2", "all"], color="#2166ac", ls=":", lw=1)
    axes[1].set_yticks(y, [""] * len(y)); axes[1].invert_yaxis(); axes[1].set_xlim(0, 108)
    axes[1].set_title("LLM judge: % correct (same 300 items)"); axes[1].set_xlabel("% judged correct")
    fig.suptitle("Style-matched pivot ties direct on official F1 and is +11 points on the judge; grey bars = translation-only ceilings")
    fig.tight_layout(); fig.savefig(OUT / "d1_systems_both_metrics.png", dpi=150); plt.close(fig)

    # d2: trade-off scatter per track
    fig, ax = plt.subplots(figsize=(8.5, 6))
    mk = {"msa": "o", "arz": "s", "ajp": "^"}
    for k in ["sft_S2", "pivot_s2", "entrack_tr", "pivot_fs"]:
        for t in ARABIC:
            ax.scatter(f1.loc[k, t], corr.loc[k, t], color=SYS[k][1], marker=mk[t], s=70, edgecolor="k", lw=.5)
        ax.scatter(f1.loc[k, "macro"], corr.loc[k, "all"], color=SYS[k][1], marker="*", s=320, edgecolor="k", lw=.7, label=SYS[k][0])
    for t in ARABIC:
        ax.scatter([], [], marker=mk[t], color="w", edgecolor="k", label=TN[t])
    ax.scatter([], [], marker="*", color="w", edgecolor="k", s=150, label="mean of 3 tracks")
    ax.set_xlabel("official F1"); ax.set_ylabel("judge: % correct"); ax.grid(alpha=.3)
    ax.set_title("Up and right is better on both metrics")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(OUT / "d2_tradeoff.png", dpi=150); plt.close(fig)

    # d3: paired differences vs direct
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = [(sn, sc) for sn in ["pivot_s2", "entrack_tr", "pivot_fs"] for sc in ["All 3 tracks", "MSA", "Egyptian", "Levantine"]]
    pr = paired.set_index(["system", "scope"]).loc[order].reset_index()
    yy = np.arange(len(pr))
    for ax, (c, lo, hi, ttl, fmt) in zip(axes, [("d_F1", "F1_lo", "F1_hi", "Official F1 minus direct (95% CI)", "{:+.3f}"),
                                                ("d_judge_pts", "judge_lo", "judge_hi", "Judge-correct minus direct, points (95% CI)", "{:+.1f}")]):
        for i, r in pr.iterrows():
            ax.errorbar(r[c], i, xerr=[[r[c] - r[lo]], [r[hi] - r[c]]], fmt="o", color=SYS[r.system][1],
                        ms=8 if r.scope == "All 3 tracks" else 5, capsize=3, mec="k")
            ax.text(r[hi], i, "  " + fmt.format(r[c]), va="center", fontsize=8)
        ax.axvline(0, color="grey"); ax.set_title(ttl, fontsize=10)
        ax.set_yticks(yy, [f"{SYS[r.system][0].replace('Pivot, ', '')} | {r.scope}" for _, r in pr.iterrows()] if ax is axes[0] else [""] * len(yy), fontsize=8)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
        for b in (3.5, 7.5):
            ax.axhline(b, color="#dddddd", lw=1)
    fig.tight_layout(); fig.savefig(OUT / "d3_paired_vs_direct.png", dpi=150); plt.close(fig)

    # d4: why F1 drops: precision, recall, ROUGE-L, both-correct items
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ks = ["sft_S2", "pivot_s2", "pivot_fs"]
    w = .26
    for i, k in enumerate(ks):
        vals = shape.loc[k, ["mm_p", "mm_r"]].values
        bars = axes[0].bar(np.arange(2) + (i - 1) * w, vals, w, color=SYS[k][1], label=SYS[k][0])
        for bb, v in zip(bars, vals):
            axes[0].text(bb.get_x() + bb.get_width() / 2, v + .001, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(range(2), ["BERTScore precision", "BERTScore recall"]); axes[0].set_ylim(.74, .80)
    axes[0].set_title("All 300 items: the loss is in precision (wording), not recall (content)"); axes[0].legend(fontsize=8)
    names = {"mm_f1": "F1", "rougeL": "ROUGE-L"}
    for i, k in enumerate(["direct", "pivot_plain", "pivot_style"]):
        vals = both_tbl.loc[["mm_f1", "rougeL"], k].values
        kk = {"direct": "sft_S2", "pivot_plain": "pivot_s2", "pivot_style": "pivot_fs"}[k]
        bars = axes[1].bar(np.arange(2) + (i - 1) * w, vals, w, color=SYS[kk][1], label=SYS[kk][0])
        for bb, v in zip(bars, vals):
            axes[1].text(bb.get_x() + bb.get_width() / 2, v + .005, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(range(2), [names["mm_f1"], names["rougeL"]])
    axes[1].set_title(f"{len(both)} items judged correct for both direct and pivot:\nsame content, lower score for plain-translator wording", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "d4_why_f1_drops.png", dpi=150); plt.close(fig)

    # d5: is the judge gain real? content judged in English before translation
    fig, ax = plt.subplots(figsize=(10, 4.6))
    series = [("Direct (answer in Arabic)", corr.loc["sft_S2", ARABIC].values, "#2166ac"),
              ("Pivot English answer, judged in English (no translation)", en_stage.values, "#9e9ac8"),
              ("Pivot after plain translation", corr.loc["pivot_s2", ARABIC].values, "#e08214"),
              ("Pivot after style-matched translation", corr.loc["pivot_fs", ARABIC].values, "#1b7837")]
    w = .2
    for i, (lb, v, c) in enumerate(series):
        bars = ax.bar(np.arange(3) + (i - 1.5) * w, v, w, color=c, label=lb)
        for bb, x in zip(bars, v):
            ax.text(bb.get_x() + bb.get_width() / 2, x + .8, f"{x:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), [TN[t] for t in ARABIC]); ax.set_ylim(0, 90); ax.set_ylabel("% judged correct")
    ax.set_title("The gain exists before any translation: the model answers better in English")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(.5, -.32))
    fig.tight_layout(); fig.savefig(OUT / "d5_judge_gain_is_content.png", dpi=150); plt.close(fig)

    # d6: verdict transitions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lbl = ["correct", "partial", "incorrect"]
    for ax, k in zip(axes, ["pivot_s2", "pivot_fs"]):
        m = trans[k].values
        ax.imshow(m, cmap="Greens")
        for a in range(3):
            for b in range(3):
                ax.text(b, a, m[a, b], ha="center", va="center", color="white" if m[a, b] > m.max() / 2 else "black")
        ax.set_xticks(range(3), lbl); ax.set_yticks(range(3), lbl)
        ax.set_xlabel(SYS[k][0]); ax.set_ylabel("Direct")
        up = int(m[1, 0] + m[2, 0] + m[2, 1]); down = int(m[0, 1] + m[0, 2] + m[1, 2])
        ax.set_title(f"{up} items improve, {down} get worse", fontsize=10)
    fig.suptitle("Judge verdict per item: direct vs pivot"); fig.tight_layout()
    fig.savefig(OUT / "d6_verdict_transitions.png", dpi=150); plt.close(fig)

    summary = {"systems": json.loads(table.to_json(orient="index")), "shape": json.loads(shape.round(4).to_json(orient="index")),
               "paired_vs_direct": paired.round(4).to_dict(orient="records"), "both_correct": json.loads(both_tbl.to_json()),
               "n_both_correct": len(both), "english_stage_correct_pct": en_stage.round(1).to_dict(),
               "transitions": {k: v.to_dict() for k, v in trans.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(table.to_string()); print(paired.round(4).to_string(index=False)); print(both_tbl.to_string()); print(en_stage)


if __name__ == "__main__":
    main()
