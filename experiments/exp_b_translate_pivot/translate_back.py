"""Step 3 (local): translate the English answers back into each track's variety and assemble prediction folders.

    python experiments/exp_b_translate_pivot/translate_back.py --kaggle results/kaggle
Reads  <kaggle>/results/predictions/pivot_en_s2/dev_<track>.jsonl   (English answers from the Kaggle step)
Writes results/assembled/predictions/pivot_s2/dev_<track>.jsonl      (the pivot system, final answers in the track's variety)
       results/assembled/predictions/pivot_en_<track>/dev_en.jsonl   (English answers, scored against the English reference)
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from translator import MODEL, Translator   # noqa: E402

ARABIC = ("msa", "arz", "ajp")


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", type=Path, default=HERE / "results" / "kaggle")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "assembled")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    pred_dir = args.kaggle / "results" / "predictions" / "pivot_en_s2"
    q_en = {(r["track"], r["id"]): r for r in read(HERE / "data" / "questions_en.jsonl")}
    tr = Translator(HERE / "data" / "translation_cache.jsonl", args.model)

    rows = {t: read(pred_dir / f"dev_{t}.jsonl") for t in ARABIC}
    items = [((t, r["id"]), t, r["answer"]) for t in ARABIC for r in rows[t]]
    back = tr.translate_many(items, "from_en")
    for t in ARABIC:
        final, english = [], []
        for r in rows[t]:
            src = q_en[(t, r["id"])]
            final.append({"system": "pivot_s2", "track": t, "split": "dev", "id": r["id"], "question": src["question"],
                          "question_en": src["question_en"], "answer_en": r["answer"], "answer": back[(t, r["id"])],
                          "raw_answer": r["raw_answer"], "fallback_used": r["fallback_used"], "error": r["error"],
                          "latency_seconds": r["latency_seconds"], "new_tokens": r["new_tokens"]})
            english.append({"system": f"pivot_en_{t}", "track": "en", "split": "dev", "id": r["id"],
                            "question": src["question_en"], "answer": r["answer"], "raw_answer": r["raw_answer"],
                            "fallback_used": r["fallback_used"], "error": r["error"]})
        for system, track, recs in ((f"pivot_s2", t, final), (f"pivot_en_{t}", "en", english)):
            d = args.out / "predictions" / system
            d.mkdir(parents=True, exist_ok=True)
            (d / f"dev_{track}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), encoding="utf-8")
    print(f"translated {len(items)} answers back with {args.model}; cost ${tr.cost:.3f}")
    for t in ARABIC:
        r = json.loads((args.out / "predictions" / "pivot_s2" / f"dev_{t}.jsonl").read_text(encoding="utf-8").splitlines()[1])
        print(t, "|", r["question_en"], "->", r["answer_en"], "->", r["answer"])


if __name__ == "__main__":
    main()
