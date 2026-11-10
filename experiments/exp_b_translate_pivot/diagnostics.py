"""Experiment B follow-up: where does the pivot lose official F1, and does the judge gain survive a fairer setup?

The dataset is parallel: every Arabic reference is a translation of the English reference of the same id. That allows
four extra systems (all on the same first 100 dev ids per Arabic track, written as prediction folders):

    ref_tr       English gold reference -> translator (plain prompt). Ceiling of "perfect English answer + this translator".
    ref_tr_fs    English gold reference -> translator shown 12 parallel examples of the reference translation style.
    pivot_fs     The pivot's English answers -> the few-shot (style-matched) translator.
    entrack_tr   sft_S2's answers on the English track (original English question, no question translation) -> plain translator.

Few-shot examples come from dev ids 101-500 (never from the 100 evaluated ids).

    python experiments/exp_b_translate_pivot/diagnostics.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from translator import LANG, MODEL, Translator   # noqa: E402

ARABIC = ("msa", "arz", "ajp")
N_SHOTS = 12

FEW_SHOT = ("Translate the final English answer into {dst}. The translations must follow the same style as these examples "
            "from the same dataset: same register, same word choices, close to the English wording and structure, same "
            "level of detail. Output only the translation of the final answer.\n\n{shots}\n\nEnglish: {text}\n{name}:")
NAME = {"msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def gold(track, n):
    rows = read(HERE / "results" / "analysis" / "files" / f"gold_{track}_{n}.jsonl")
    return {r["id"]: r["answer"][track] for r in rows}, [r["id"] for r in rows]


def write(system, track, recs, out):
    d = out / "predictions" / system
    d.mkdir(parents=True, exist_ok=True)
    (d / f"dev_{track}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "diagnostics")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    tr = Translator(HERE / "data" / "translation_cache.jsonl", args.model)

    en100, ids100 = gold("en", 100)
    en500, ids500 = gold("en", 500)
    pool = [i for i in ids500[100:] if 12 <= len(en500[i].split()) <= 40]
    shots_ids = random.Random(0).sample(pool, N_SHOTS)
    shots = {}
    for t in ARABIC:
        tg, _ = gold(t, 500)
        shots[t] = "\n\n".join(f"English: {en500[i]}\n{NAME[t]}: {tg[i]}" for i in shots_ids)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "few_shot_ids.json").write_text(json.dumps(shots_ids, indent=1))

    def fs_prompt(track, text):
        return FEW_SHOT.format(dst=LANG[track], shots=shots[track], text=text, name=NAME[track])

    pivot = {t: {r["id"]: r for r in read(HERE / "results" / "assembled" / "predictions" / "pivot_s2" / f"dev_{t}.jsonl")}
             for t in ARABIC}
    entrack = {r["id"]: r["answer"] for r in read(REPO / "results" / "phase1" / "full" / "results" / "predictions" / "sft_S2" / "dev_en.jsonl")}

    jobs = {  # system -> (source English text per id, direction label, template)
        "ref_tr": (lambda t, i: en100[i], "from_en", None),
        "ref_tr_fs": (lambda t, i: en100[i], "from_en_fs12", fs_prompt),
        "pivot_fs": (lambda t, i: pivot[t][i]["answer_en"], "from_en_fs12", fs_prompt),
        "entrack_tr": (lambda t, i: entrack[i], "from_en", None),
    }
    for system, (src, direction, template) in jobs.items():
        items = [((t, i), t, src(t, i)) for t in ARABIC for i in ids100]
        back = tr.translate_many(items, direction, template)
        for t in ARABIC:
            write(system, t, [{"system": system, "track": t, "split": "dev", "id": i, "source_en": src(t, i),
                               "answer": back[(t, i)]} for i in ids100], args.out)
        print(f"{system}: {len(items)} translations")
    print(f"total new cost ${tr.cost:.3f}")


if __name__ == "__main__":
    main()
