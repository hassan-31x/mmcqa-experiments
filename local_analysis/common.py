"""Shared loading and statistics for the local analysis. numpy only; pandas is imported lazily."""

import json
import sys
from pathlib import Path

import numpy as np

from .paths import REPO, ROWS

sys.path.insert(0, str(REPO))
from mmcqa_exp.util import read_jsonl, write_jsonl   # noqa: E402,F401  (re-exported)

TRACKS = ("en", "msa", "arz", "ajp")
TRACK_NAMES = {"en": "English", "msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}
SOURCES = ("predictions", "results/predictions", "candidates", "selection")


def gold_rows() -> dict:
    """Dev gold rows per track: ``id, image, question, answer`` (1,000 each, same order across tracks)."""
    out = {}
    for t in TRACKS:
        path = ROWS / f"dev_{t}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Dev gold rows are expected under external/mmcqa_rows/.")
        out[t] = read_jsonl(path)
    return out


def eval_ids(n: int) -> list:
    return [r["id"] for r in gold_rows()["en"][:n]]


def discover_predictions(dirs) -> tuple:
    """Find ``<dir>/<source>/<system>/dev_<track>.jsonl``. Returns ``(preds, origin)``.

    ``preds[system][track][id] = record``; ``origin[system]`` is the source folder it came from.
    """
    preds, origin = {}, {}
    for d in dirs:
        for source in SOURCES:
            root = Path(d) / source
            if not root.is_dir():
                continue
            for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for f in sorted(sys_dir.glob("dev_*.jsonl")):
                    track = f.stem.split("_", 1)[1]
                    if track not in TRACKS:
                        continue
                    rows = read_jsonl(f)
                    if rows:
                        preds.setdefault(sys_dir.name, {})[track] = {r["id"]: r for r in rows}
                        origin[sys_dir.name] = source.split("/")[-1]
    return preds, origin


def load_per_item(path):
    import pandas as pd
    rows = read_jsonl(path)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def score_matrix(df, system: str, ids: list, col: str = "mm_f1") -> np.ndarray:
    """``(len(ids), 4)`` array of per-item scores for one system (NaN where missing)."""
    sub = df[df.system == system]
    out = np.full((len(ids), len(TRACKS)), np.nan)
    pos = {i: k for k, i in enumerate(ids)}
    for j, t in enumerate(TRACKS):
        s = sub[sub.track == t]
        for i, v in zip(s.id, s[col]):
            if i in pos:
                out[pos[i], j] = v
    return out


def macro_ci(F: np.ndarray, boot: int = 2000, seed: int = 0) -> tuple:
    """Macro mean and 95% bootstrap CI, resampling ids (the four tracks of an id stay together)."""
    F = F[~np.isnan(F).any(axis=1)]
    if len(F) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(F), (boot, len(F)))
    means = F[idx].mean(axis=(1, 2))
    return float(F.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(Fa: np.ndarray, Fb: np.ndarray, boot: int = 4000, seed: int = 0) -> dict:
    """Macro difference a-b on the ids both have, with a paired bootstrap CI and two-sided p."""
    keep = ~(np.isnan(Fa).any(axis=1) | np.isnan(Fb).any(axis=1))
    D = (Fa - Fb)[keep]
    if len(D) == 0:
        return {"n": 0, "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "win": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), (boot, len(D)))
    means = D[idx].mean(axis=(1, 2))
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": int(len(D)), "diff": float(D.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p": float(max(p, 1.0 / boot)),
            "win": float((D.mean(axis=1) > 0).mean())}


def per_track_means(F: np.ndarray) -> list:
    return [float(np.nanmean(F[:, j])) if (~np.isnan(F[:, j])).any() else float("nan") for j in range(F.shape[1])]


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str))
