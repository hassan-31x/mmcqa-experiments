"""Tables, charts, significance tests and the decision, written to results/report/phase_report.md.

    python -m local_analysis.compare --analysis results/analysis/NAME --phase1 results/phase1/TAG \
        --phase2 results/phase2/TAG [--expect-rows 500] [--report results/report]

Everything is read from files written by score / ensemble_select / judge; nothing is recomputed with a model.
The decision rules are the ones fixed in plans/README.md before any result was seen.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (TRACKS, TRACK_NAMES, eval_ids, load_per_item, macro_ci, paired_diff, per_track_means,
                     read_jsonl, score_matrix)
from .paths import BASELINE_SCORES, HISTORY_SCORES, RESULTS

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a86", "#0b0b0b", "#52514e"   # dataviz reference palette
HISTORY_LABELS = {"hist_culturalpangea_7b": "CulturalPangea-7B (old run)", "hist_gemma_4_12b": "Gemma-4-12B (old run)",
                  "hist_qwen3_5_4b": "Qwen3.5-4B (old run)", "hist_qwen3_5_9b": "Qwen3.5-9B (old run)",
                  "hist_official_baseline": "Official baseline (dev100)", "hist_echo_question": "Echo the question (dev100)",
                  "full_official_baseline": "Official baseline (full-dev run)", "full_baseline_trimmed": "Baseline, invented turns trimmed",
                  "full_echo_question": "Echo the question", "full_tfidf_retrieval": "TF-IDF train retrieval"}
EXPECTED_HISTORY = {"hist_culturalpangea_7b": 0.730, "hist_gemma_4_12b": 0.721, "hist_qwen3_5_4b": 0.718,
                    "hist_qwen3_5_9b": 0.704, "hist_official_baseline": 0.690, "hist_echo_question": 0.667}
EGY = {"ده", "دي", "دول", "كده", "كدا", "إزاي", "ازاي", "إيه", "ايه", "عشان", "علشان", "بتاع", "بتاعة", "بتاعت", "أوي", "اوي",
       "فين", "دلوقتي", "باين", "باينة", "بيورّي", "بتورّي", "بيوري", "بتوري", "هيا"}
LEV = {"شو", "هيك", "هاد", "هادا", "هاي", "هلق", "هلأ", "منيح", "كتير", "عم", "وين", "ليش", "بدو", "بدها", "هون", "هنيك",
       "مبين", "مبينة", "مبينين", "متل", "لهيك", "هيدا", "هيدي", "بيبين", "بتبين"}
COLLOQUIAL = {"اللي", "مش", "زي", "كمان", "بس", "إللي"}
STRIP = "؟?!.,،:;\"'()[]«»*-–—"


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and v == v else ("" if v != v else str(v))
                 for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- data
def load_everything(analysis: Path, ids500: list):
    new = pd.concat([d for d in (load_per_item(analysis / "per_item_scores.jsonl"),
                                 load_per_item(analysis / "selection_scores.jsonl")) if not d.empty], ignore_index=True)
    parts = [new]
    if HISTORY_SCORES.exists():
        hist = pd.DataFrame(read_jsonl(HISTORY_SCORES))
        hist["system"] = "hist_" + hist.system
        hist["words"] = hist.prediction.str.split().str.len()
        parts.append(hist)
    if BASELINE_SCORES.exists():
        base = pd.DataFrame(read_jsonl(BASELINE_SCORES))
        base = base[base.stage == "dev"].copy()
        base["system"] = "full_" + base.system
        base["words"] = base.prediction.str.split().str.len()
        parts.append(base)
    cols = ["system", "track", "id", "question", "reference", "prediction", "mm_p", "mm_r", "mm_f1", "words",
            "latency_seconds", "fallback_used", "mean_logprob"]
    return pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)


def is_new(system: str) -> bool:
    return not system.startswith(("hist_", "full_"))


def system_table(df, systems, ids, labels=None) -> pd.DataFrame:
    rows = []
    for s in systems:
        F = score_matrix(df, s, ids)
        if np.isnan(F).all():
            continue
        m, lo, hi = macro_ci(F)
        sub = df[(df.system == s) & df.id.isin(set(ids))]
        ref_words = sub.reference.str.split().str.len().mean()
        rows.append({"system": (labels or {}).get(s, s), **{TRACK_NAMES[t]: v for t, v in zip(TRACKS, per_track_means(F))},
                     "macro F1": m, "95% CI": f"[{lo:.4f}, {hi:.4f}]", "precision": sub.mm_p.mean(), "recall": sub.mm_r.mean(),
                     "words": sub.words.mean(), "len ratio": sub.words.mean() / ref_words if ref_words else np.nan,
                     "s/item": sub.latency_seconds.mean(), "fallback %": 100 * sub.fallback_used.fillna(False).astype(float).mean(),
                     "n": int(sub.shape[0] // len(TRACKS)), "_macro": m})
    return pd.DataFrame(rows).sort_values("_macro", ascending=False).drop(columns="_macro")


def dialect_rate(texts, track):
    flag = EGY if track == "arz" else LEV
    hit = 0
    for t in texts:
        w = {x.strip(STRIP) for x in str(t).split()}
        hit += bool(w & flag) or bool(w & COLLOQUIAL)
    return 100 * hit / max(len(texts), 1)


# ----------------------------------------------------------------------------- charts
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    return plt


def bar_chart(rows, path, title, xlabel="macro official F1 (mmBERT layer 17)"):
    """rows: [(label, value, lo, hi, kind)]; kind in new|pangea|other. Values are printed on the bars."""
    if not rows:
        return
    plt = setup_mpl()
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4))
    colors = {"new": BLUE, "pangea": ORANGE, "other": GRAY}
    for k, (label, v, lo, hi, kind) in enumerate(rows):
        ax.barh(k, v, color=colors[kind], height=0.62)
        if lo == lo:
            ax.plot([lo, hi], [k, k], color=INK, lw=1.2)
        ax.text(max(hi if hi == hi else v, v) + 0.0015, k, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    lo_lim = min(r[1] for r in rows) - 0.05
    ax.set_xlim(max(0.6, lo_lim), max(r[3] if r[3] == r[3] else r[1] for r in rows) + 0.02)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.012, "blue = new systems, orange = CulturalPangea, gray = earlier runs and baselines. "
             "The axis is truncated: compare values, not bar lengths.", ha="left", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pr_scatter(table, path):
    if table.empty:
        return
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for _, r in table.iterrows():
        kind = ORANGE if "pangea" in r.system.lower() else BLUE if is_new(r.system.split(" (")[0]) else GRAY
        ax.scatter(r.recall, r.precision, s=46, color=kind, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.annotate(r.system, (r.recall, r.precision), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    lim = [min(table.recall.min(), table.precision.min()) - 0.01, max(table.recall.max(), table.precision.max()) + 0.01]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("BERTScore recall")
    ax.set_ylabel("BERTScore precision")
    ax.set_title("Precision vs recall (above the dashed line = answers shorter than the references)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sft_curves(phase1: Path, path):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    drawn = False
    for k, name in enumerate(("S1", "S2")):
        hist = phase1 / "results" / "logs" / f"sft_{name}_history.csv"
        if not hist.exists():
            continue
        h = pd.read_csv(hist)
        color = (BLUE, ORANGE)[k]
        tr, ev = h.dropna(subset=["loss"]), h.dropna(subset=["eval_loss"])
        axes[0].plot(tr.step, tr.loss, color=color, lw=1.6)
        axes[0].text(tr.step.iloc[-1], tr.loss.iloc[-1], f" {name}", color=INK, fontsize=9, va="center")
        axes[1].plot(ev.step, ev.eval_loss, color=color, lw=1.6, marker="o", ms=4)
        axes[1].text(ev.step.iloc[-1], ev.eval_loss.iloc[-1], f" {name}", color=INK, fontsize=9, va="center")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    axes[0].set_title("training loss (answer tokens)", loc="left", fontsize=10)
    axes[1].set_title("eval loss (200 held-out dev items)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- decision
def decide(df, ids, judge, selection) -> dict:
    out = {"notes": []}
    have = set(df.system.unique())
    sfts = [s for s in have if s.startswith("sft_") and s != "sft_SMOKE" and not s.startswith("sft_SMOKE")]
    if not {"qwen_z1", "pangea_p0"} <= have or not sfts:
        out["notes"].append("Decision needs pangea_p0, qwen_z1 and at least one sft_* system.")
        return out
    F = {s: score_matrix(df, s, ids) for s in [*sfts, "qwen_z1", "pangea_p0"]}
    best = max(sfts, key=lambda s: np.nanmean(F[s]))
    out["best_sft"] = best
    train = paired_diff(F[best], F["qwen_z1"])
    out["fine_tuning_works"] = bool(train["lo"] > 0)
    out["sft_minus_z1"] = train
    vs = paired_diff(F[best], F["pangea_p0"])
    out["sft_minus_pangea"] = vs
    per_track = [float(np.nanmean(F[best][:, j] - F["pangea_p0"][:, j])) for j in range(4)]
    out["sft_minus_pangea_per_track"] = dict(zip(TRACKS, per_track))
    jq = {}
    if judge is not None and not judge.empty:
        jq = {r.system: r.correct_pct for r in judge.itertuples()}
    j_sft, j_pg = jq.get(best), jq.get("pangea_p0")
    out["judge_correct_pct"] = {"sft": j_sft, "pangea": j_pg}
    judge_ok = None if j_sft is None or j_pg is None else j_sft >= j_pg
    clearly_better = vs["diff"] >= 0.005 and vs["lo"] > 0 and min(per_track) >= -0.005
    tied = abs(vs["diff"]) < 0.005
    if clearly_better and judge_ok is not False:
        out["single_model"] = best
        out["single_reason"] = "beats Pangea by >= 0.005 (CI > 0), no track worse by > 0.005" + (
            " and judge correctness is at least Pangea's" if judge_ok else " (judge not run: confirm before final)")
    elif tied and judge_ok:
        out["single_model"] = best
        out["single_reason"] = "tied with Pangea on F1 but clearly better on judge correctness"
    elif clearly_better and judge_ok is False:
        out["single_model"] = "pangea_p0"
        out["single_reason"] = "F1 win for the fine-tuned model is not matched by judge correctness (likely style gaming)"
    else:
        out["single_model"] = "pangea_p0"
        out["single_reason"] = "the fine-tuned model does not clear the rule (>= 0.005 with CI > 0, or tie + judge win)"
    if selection is not None and not selection.empty:
        singles = {s: float(np.nanmean(F[s])) for s in (best, "pangea_p0", "qwen_z1")}
        best_single = max(singles, key=singles.get)
        cand = selection[(selection.method != "oracle") & ~selection.method.str.startswith("single")
                         & ~selection.method.isin(["per_track_best"])]          # only honest (cross-validated or fit-free) methods
        if not cand.empty:
            top = cand.sort_values("macro", ascending=False).iloc[0]
            ref = selection[(selection.pool == top.pool) & (selection.method == f"single:{best_single}")]
            ref_macro = float(ref.macro.iloc[0]) if not ref.empty else singles[best_single]
            gain = float(top.macro) - ref_macro
            out["ensemble"] = {"method": top.method, "pool": top.pool, "macro": float(top.macro), "ci_lo": float(top.ci_lo),
                               "best_single": best_single, "best_single_macro": ref_macro, "gain": gain,
                               "adopt": bool(gain >= 0.003 and top.ci_lo > ref_macro)}
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--phase1", type=Path)
    ap.add_argument("--phase2", type=Path)
    ap.add_argument("--expect-rows", type=int, default=500)
    ap.add_argument("--report", type=Path, default=RESULTS / "report")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args(argv)
    ids500, ids100 = eval_ids(args.expect_rows), eval_ids(100)
    report, tables, charts = args.report, args.report / "tables", args.report / "charts"
    tables.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    df = load_everything(args.analysis, ids500)
    all_new = sorted(s for s in df.system.unique() if is_new(s))
    new_systems = [s for s in all_new if not s.startswith("sel_")]       # selection ensembles have their own section
    judge = pd.read_csv(args.analysis / "judge" / "judge_summary.csv") if (args.analysis / "judge" / "judge_summary.csv").exists() else None
    selection = pd.read_csv(args.analysis / "selection_summary.csv") if (args.analysis / "selection_summary.csv").exists() else None
    md = ["# Phase 1 and 2 report (auto-generated by local_analysis.compare)\n"]

    # 1. dev100 history table
    hist_systems = [s for s in HISTORY_LABELS if s.startswith("hist_") and s in set(df.system)]
    t1 = system_table(df, [*hist_systems, *new_systems], ids100, HISTORY_LABELS)
    t1.to_csv(tables / "01_dev100_history.csv", index=False)
    md += ["## 1. Comparison with every earlier run (dev100: the 100 items per track they all answered)\n", md_table(t1.drop(columns=["s/item", "fallback %"])),
           ""]
    for s, expected in EXPECTED_HISTORY.items():
        if s in set(df.system):
            got = float(np.nanmean(score_matrix(df, s, ids100)))
            if abs(got - expected) > 0.002:
                md.append(f"> WARNING: {s} macro {got:.4f} differs from the published {expected:.3f}\n")

    # 2. dev500 table
    base_systems = [s for s in ("full_official_baseline", "full_baseline_trimmed", "full_echo_question", "full_tfidf_retrieval")
                    if s in set(df.system)]
    t2 = system_table(df, [*new_systems, *base_systems], ids500, HISTORY_LABELS)
    t2.to_csv(tables / "02_dev500_main.csv", index=False)
    md += [f"## 2. Main table (dev500: first {args.expect_rows} dev items per track)\n", md_table(t2), ""]
    rows = []
    for s in [*new_systems, *base_systems]:
        F = score_matrix(df, s, ids500)
        if np.isnan(F).any():
            continue
        m, lo, hi = macro_ci(F)
        kind = "pangea" if s in ("pangea_p0", "cascade_qwen2pangea") else ("new" if is_new(s) else "other")
        rows.append((HISTORY_LABELS.get(s, s), m, lo, hi, kind))
    bar_chart(rows, charts / "01_dev500_macro_f1.png", f"Official F1 on dev{args.expect_rows} (macro over four tracks, 95% CI)")
    pr_scatter(t2.assign(system=t2.system), charts / "02_precision_recall.png")

    # 3. gain decomposition for Qwen
    steps = []
    if {"hist_qwen3_5_4b", "qwen_z1"} <= set(df.system):
        steps.append(("old Qwen3.5-4B -> qwen_z1 (new prompt, cleanup, 448px)", "hist_qwen3_5_4b", "qwen_z1", ids100))
    sfts = sorted(s for s in new_systems if s.startswith("sft_") and "SMOKE" not in s)
    for s in sfts:
        if "qwen_z1" in set(df.system):
            steps.append((f"qwen_z1 -> {s} (training)", "qwen_z1", s, ids500))
    if {"hist_culturalpangea_7b", "pangea_p0"} <= set(df.system):
        steps.append(("old Pangea run -> pangea_p0 (reproduction check)", "hist_culturalpangea_7b", "pangea_p0", ids100))
    dec_rows = []
    for label, a, b, ids in steps:
        d = paired_diff(score_matrix(df, b, ids), score_matrix(df, a, ids))
        dec_rows.append({"step": label, "n ids": d["n"], "delta macro F1": d["diff"], "95% CI": f"[{d['lo']:.4f}, {d['hi']:.4f}]",
                         "p": d["p"], "new wins %": 100 * d["win"]})
    t3 = pd.DataFrame(dec_rows)
    t3.to_csv(tables / "03_gain_decomposition.csv", index=False)
    md += ["## 3. Where the gains come from\n", md_table(t3), ""]

    # 4. pairwise vs pangea_p0 and qwen_z1
    pw = []
    for ref in ("pangea_p0", "qwen_z1"):
        if ref not in set(df.system):
            continue
        for s in new_systems:
            if s == ref or s.startswith("sel_") and False:
                continue
            d = paired_diff(score_matrix(df, s, ids500), score_matrix(df, ref, ids500))
            if d["n"]:
                pw.append({"system": s, "vs": ref, "delta macro F1": d["diff"], "95% CI": f"[{d['lo']:.4f}, {d['hi']:.4f}]",
                           "p": d["p"], "wins %": 100 * d["win"]})
    t4 = pd.DataFrame(pw)
    t4.to_csv(tables / "04_pairwise.csv", index=False)
    md += ["## 4. Paired tests (dev500)\n", md_table(t4), ""]

    # 5. selection ensembles
    if selection is not None:
        sel = selection.sort_values(["pool", "macro"], ascending=[True, False]).copy()
        sel["honest"] = sel.method.map(lambda m: "upper bound" if m == "oracle" else ("fit on all items (optimistic)" if m == "per_track_best"
                                                                                      else "reference" if m.startswith("single") else "cross-validated / fit-free"))
        sel["95% CI"] = sel.apply(lambda r: f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]", axis=1)
        t5 = sel[["pool", "method", "macro", "95% CI", *[f"f1_{t}" for t in TRACKS], "honest", "pool_systems"]]
        t5.to_csv(tables / "05_selection.csv", index=False)
        md += ["## 5. Selection ensembles (no extra GPU)\n", md_table(t5),
               "Only cross-validated or fit-free rows count towards decisions. The oracle is an upper bound.\n"]
        top = [(f"{r.method} [{r.pool}]", r.macro, r.ci_lo, r.ci_hi, "new" if not r.method.startswith("single") else "other")
               for r in sel.itertuples() if r.method != "oracle"][:14]
        bar_chart(top, charts / "03_selection.png", "Selection ensembles vs single systems (same items per pool)")

    # 6. judge
    if judge is not None:
        md += ["## 6. LLM judge (first 100 dev ids per track; verdicts against the reference answer)\n", md_table(judge, "{:.2f}"),
               "`f1_vs_correct_auc` is how well mmBERT F1 separates correct from not-correct answers for that system.\n"]

    # 7. dialect
    drows = []
    for s in [*new_systems, *hist_systems]:
        r = {"system": HISTORY_LABELS.get(s, s)}
        for t in ("arz", "ajp"):
            sub = df[(df.system == s) & (df.track == t) & df.id.isin(set(ids500))]
            r[f"{TRACK_NAMES[t]} dialect-marker %"] = dialect_rate(sub.prediction, t) if len(sub) else np.nan
        drows.append(r)
    gold = df[(df.system == new_systems[0]) if new_systems else df.system.notna()]
    g = {t: dialect_rate(gold[gold.track == t].drop_duplicates("id").reference, t) for t in ("arz", "ajp")}
    drows.append({"system": "gold references", **{f"{TRACK_NAMES[t]} dialect-marker %": g[t] for t in ("arz", "ajp")}})
    t7 = pd.DataFrame(drows)
    t7.to_csv(tables / "07_dialect.csv", index=False)
    md += ["## 7. Dialect markers (lexical heuristic; short answers contain fewer markers)\n", md_table(t7, "{:.1f}")]

    # 8. SFT training
    if args.phase1:
        srows = []
        for p in sorted((args.phase1 / "results" / "logs").glob("sft_*_summary.json")):
            j = json.loads(p.read_text())
            srows.append({"config": j["config"]["name"], "lr": j["config"]["lr"], "steps": j["steps_done"],
                          "fraction of epoch": j["fraction_of_epoch"], "s/step": j["sec_per_step"], "batch": f"{j['per_device_bs']}x{j['grad_accum']}",
                          "best eval loss": j["best_eval_loss"], "best step": j["best_step"], "hours": j["hours"],
                          "peak mem GB": j["peak_mem_gb"], "stop": j["stop_reason"]})
        t8 = pd.DataFrame(srows)
        t8.to_csv(tables / "08_sft_training.csv", index=False)
        md += ["## 8. Fine-tuning runs\n", md_table(t8), "![curves](charts/04_sft_curves.png)\n"]
        sft_curves(args.phase1, charts / "04_sft_curves.png")

    # 9. cost
    comp = {"cascade_qwen2pangea": ["qwen_z1", "cascade_qwen2pangea"], "fusion_qwen": ["pangea_p0", "qwen_z1", "fusion_qwen"]}
    if selection is not None:
        for r in selection.itertuples():
            comp[f"sel_{r.method}_{r.pool}"] = str(r.pool_systems).split(",")
    lat = df[df.system.isin(new_systems)].groupby("system").latency_seconds.mean().to_dict()
    crows = []
    for s in all_new:
        parts = comp.get(s, [s])
        sec = sum((lat.get(p) or 0.0) for p in parts)
        if sec:
            crows.append({"system": s, "components": "+".join(parts), "s/item (single worker)": sec,
                          "GPU-hours for 4,000 test items": sec * 4000 / 3600})
    t9 = pd.DataFrame(crows)
    t9.to_csv(tables / "09_cost.csv", index=False)
    md += ["## 9. Inference cost for a 4,000-item test set (one worker, T4)\n", md_table(t9, "{:.2f}")]

    # decision
    dec = decide(df, ids500, judge, selection)
    (report / "decision.json").write_text(json.dumps(dec, indent=1, default=str))
    md += ["## Decision (rules fixed in plans/README.md)\n"]
    if "single_model" in dec:
        s = dec["sft_minus_z1"]
        md += [f"- **Does fine-tuning work?** {'Yes' if dec['fine_tuning_works'] else 'Not shown'}: `{dec['best_sft']}` minus `qwen_z1` = "
               f"{s['diff']:+.4f} (95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]).",
               f"- **Fine-tuned Qwen vs Pangea:** {dec['sft_minus_pangea']['diff']:+.4f} (CI [{dec['sft_minus_pangea']['lo']:+.4f}, "
               f"{dec['sft_minus_pangea']['hi']:+.4f}]); per track "
               + ", ".join(f"{t} {v:+.4f}" for t, v in dec["sft_minus_pangea_per_track"].items())
               + f". Judge correct: {dec['judge_correct_pct']}.",
               f"- **Single model to use:** `{dec['single_model']}` ({dec['single_reason']})."]
        if "ensemble" in dec:
            e = dec["ensemble"]
            md.append(f"- **Ensemble:** best honest method `{e['method']}` on pool `{e['pool']}` = {e['macro']:.4f} vs best single "
                      f"`{e['best_single']}` {e['best_single_macro']:.4f} (gain {e['gain']:+.4f}). "
                      f"{'Worth adopting' if e['adopt'] else 'Not adopted (needs >= +0.003 with CI above the single system)'}; "
                      "also check the judge correct rate does not drop by more than 3 points.")
    else:
        md += ["- " + n for n in dec["notes"]]
    (report / "phase_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # experiments.csv
    exp_path = report.parent / "experiments.csv"
    jmap = {r.system: r for r in judge.itertuples()} if judge is not None else {}
    existing = list(csv.DictReader(exp_path.open())) if exp_path.exists() else []
    seen = {(r["system"], r["run_tag"]) for r in existing}
    fields = ["system", "run_tag", "en", "msa", "arz", "ajp", "macro", "macro_ci_lo", "macro_ci_hi", "eval_items",
              "judge_correct_pct", "judge_variety_ok_pct", "s_per_item"]
    with exp_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not existing:
            w.writeheader()
        for s in all_new:
            if (s, args.run_tag) in seen:
                continue
            F = score_matrix(df, s, ids500)
            if np.isnan(F).any():
                continue
            m, lo, hi = macro_ci(F)
            j = jmap.get(s)
            w.writerow({"system": s, "run_tag": args.run_tag, **dict(zip(TRACKS, per_track_means(F))), "macro": m,
                        "macro_ci_lo": lo, "macro_ci_hi": hi, "eval_items": len(ids500) * 4,
                        "judge_correct_pct": getattr(j, "correct_pct", ""), "judge_variety_ok_pct": getattr(j, "dialect_variety_ok_pct", ""),
                        "s_per_item": lat.get(s, "")})
    print(f"[compare] wrote {report / 'phase_report.md'}")
    print("\n".join(md[-8:]))


if __name__ == "__main__":
    main()
