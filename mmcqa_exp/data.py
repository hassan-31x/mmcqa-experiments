"""Rows, images and the official starter kit, all at pinned revisions.

Network imports are lazy so that workers (which only need ``util`` and a backend) can run in an
environment where a different ``huggingface_hub`` is installed.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

from . import settings as S
from .util import free_disk_gb, read_jsonl


def load_rows(split: str, track: str) -> list:
    """All rows of ``qa/mena/{split}_{track}.jsonl``: ``id, image, question`` (+ ``answer`` unless devtest)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(S.DATASET_REPO, f"{S.TASK}/{S.REGION}/{split}_{track}.jsonl",
                           repo_type="dataset", revision=S.DATASET_REVISION)
    return read_jsonl(Path(path))


def eval_rows(split: str = S.EVAL_SPLIT, n: int = S.EVAL_ROWS) -> dict:
    """The first ``n`` rows of every track. Row order is identical across tracks, so ids are shared."""
    out = {t: load_rows(split, t)[:n] for t in S.TRACKS}
    check_shared_ids(out)
    return out


def eval_loss_rows() -> dict:
    lo, hi = S.EVAL_LOSS_SLICE
    out = {t: load_rows("dev", t)[lo:hi] for t in S.TRACKS}
    check_shared_ids(out)
    return out


def train_rows(limit=None) -> dict:
    out = {t: load_rows("train", t) for t in S.TRACKS}
    if limit:
        out = {t: rows[:limit] for t, rows in out.items()}
    return out


def check_shared_ids(rows_by_track: dict) -> None:
    first = [r["id"] for r in rows_by_track[S.TRACKS[0]]]
    for t in S.TRACKS[1:]:
        if [r["id"] for r in rows_by_track[t]] != first:
            raise RuntimeError(f"Track {t} does not list the same ids in the same order as {S.TRACKS[0]}")


def images_needed(rows_by_track: dict) -> set:
    return {r["image"] for rows in rows_by_track.values() for r in rows}


def fetch_images(split: str, media_root, needed) -> None:
    """Download the split's image archive and unpack just the images the run needs."""
    from huggingface_hub import hf_hub_download
    media_root = Path(media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(needed) if not (media_root / p).is_file()]
    print(f"[data] {split}: {len(todo)} images to unpack; free disk {free_disk_gb(media_root):.1f} GB", flush=True)
    if not todo:
        return
    archive = hf_hub_download(S.DATASET_REPO, f"archives/images_{S.REGION}_{split}.zip",
                              repo_type="dataset", revision=S.DATASET_REVISION)
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        for name in todo:
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe image path: {name}")
            if name in members:
                zf.extract(name, media_root)
    missing = [p for p in todo if not (media_root / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from images_{S.REGION}_{split}.zip, e.g. {missing[0]}")
    print(f"[data] {split}: {len(todo)} images unpacked under {media_root}", flush=True)


def official_repo(workdir) -> Path:
    """Clone the organisers' starter kit at the pinned commit (format checker, scorer)."""
    path = Path(workdir) / "mmcultureqa"
    if not (path / ".git").exists():
        subprocess.run(["git", "clone", "-q", S.OFFICIAL_REPO_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", S.OFFICIAL_COMMIT], check=True)
    return path


def python_exe_of(env_dir) -> str:
    return str(Path(env_dir) / "bin" / "python") if env_dir else sys.executable
