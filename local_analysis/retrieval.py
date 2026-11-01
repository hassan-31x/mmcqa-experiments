"""TF-IDF train-retrieval candidate: the reference answer of the most similar train question (no model, no image).

    python -m local_analysis.retrieval --out results/analysis/NAME [--rows 500]

Reproduces mmcqa_baseline.score.Retriever (char 2-4-gram TF-IDF). Writes candidates/tfidf_retrieval/.
Needs scikit-learn and huggingface_hub; works in either local environment.
"""

import argparse
from pathlib import Path

from .common import TRACKS, gold_rows, read_jsonl, write_jsonl
from .paths import REPO


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=500)
    args = ap.parse_args(argv)
    from huggingface_hub import hf_hub_download
    from sklearn.feature_extraction.text import TfidfVectorizer
    from mmcqa_exp import settings as S

    gold = gold_rows()
    for t in TRACKS:
        path = hf_hub_download(S.DATASET_REPO, f"qa/mena/train_{t}.jsonl", repo_type="dataset",
                               revision=S.DATASET_REVISION)
        train = read_jsonl(Path(path))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
        matrix = vec.fit_transform([r["question"] for r in train])
        dev = gold[t][:args.rows]
        sims = (vec.transform([r["question"] for r in dev]) @ matrix.T).toarray()
        rows = []
        for r, s in zip(dev, sims):
            pick = int(s.argmax())
            rows.append({"system": "tfidf_retrieval", "track": t, "split": "dev", "id": r["id"],
                         "question": r["question"], "answer": train[pick]["answer"],
                         "raw_answer": train[pick]["answer"], "fallback_used": False, "error": None,
                         "meta": {"train_id": train[pick]["id"], "similarity": float(s[pick])}})
        write_jsonl(args.out / "candidates" / "tfidf_retrieval" / f"dev_{t}.jsonl", rows)
        print(f"tfidf_retrieval {t}: {len(rows)} items", flush=True)


if __name__ == "__main__":
    main()
