"""Step 1 (local): translate the first N dev questions of the three Arabic tracks into English.

    python experiments/exp_b_translate_pivot/translate_questions.py --rows 100
Writes data/questions_en.jsonl (id, track, question, question_en) and data/translation_cache.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from translator import MODEL, Translator   # noqa: E402

sys.path.insert(0, str(HERE.parents[1]))
from local_analysis.common import gold_rows   # noqa: E402

ARABIC = ("msa", "arz", "ajp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    gold = gold_rows()
    tr = Translator(data / "translation_cache.jsonl", args.model)
    items = [((t, r["id"]), t, r["question"]) for t in ARABIC for r in gold[t][:args.rows]]
    out = tr.translate_many(items, "to_en")
    with (data / "questions_en.jsonl").open("w", encoding="utf-8") as fh:
        for t in ARABIC:
            for r in gold[t][:args.rows]:
                fh.write(json.dumps({"id": r["id"], "track": t, "question": r["question"],
                                     "question_en": out[(t, r["id"])]}, ensure_ascii=False) + "\n")
    print(f"translated {len(items)} questions with {args.model}; cost ${tr.cost:.3f}")
    for t in ARABIC:
        for r in gold[t][:2]:
            print(t, "|", r["question"], "->", out[(t, r["id"])])


if __name__ == "__main__":
    main()
