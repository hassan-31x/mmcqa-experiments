"""Where things live on the Mac. Everything is relative to this repo's parent (data-processing/)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent                                  # .../data-processing
EXTERNAL = PROJECT.parent / "external"                 # .../nlp/project/external
OFFICIAL = EXTERNAL / "mmcultureqa"                    # organisers' starter kit (scorer, format checker)
ROWS = EXTERNAL / "mmcqa_rows"                         # dev_{en,msa,arz,ajp}.jsonl, 1000 rows each
SCORER_PY = EXTERNAL / ".venv-scorer" / "bin" / "python"
HISTORY_SCORES = PROJECT / "analysis" / "official" / "scores" / "per_item_scores.jsonl"     # dev100, official scorer
BASELINE_SCORES = PROJECT / "mmcqa-baseline" / "results" / "scores" / "per_item_scores.jsonl"   # full dev, official scorer
RESULTS = REPO / "results"
HF_SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots"
