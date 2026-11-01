"""SFT examples: [image + question] -> reference answer."""

import random
from pathlib import Path

from .. import prompts


class ListDataset:
    """Map-style dataset over a list (works with torch DataLoader without importing torch here)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_examples(rows_by_track: dict, media, prompt_fn=prompts.p_new, shuffle_seed=None):
    """Return ``(examples, n_skipped)``. Tracks are mixed; rows without an answer are skipped."""
    media = Path(media)
    examples, skipped = [], 0
    for track, rows in rows_by_track.items():
        for r in rows:
            answer = str(r.get("answer") or "").strip()
            if not answer:
                skipped += 1
                continue
            examples.append({"id": r["id"], "track": track, "image": str(media / r["image"]),
                             "prompt": prompt_fn(r["question"], track), "answer": answer[:1500]})
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(examples)
    return examples, skipped
