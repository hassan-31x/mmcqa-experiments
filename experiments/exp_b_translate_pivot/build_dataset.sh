#!/bin/bash
# Build and upload the private Kaggle dataset with the S2 adapter and the translated questions.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/data/kaggle_dataset"
rm -rf "$OUT" && mkdir -p "$OUT/adapters/S2"
cp "$REPO"/results/phase1/full/results/adapters/S2/adapter_config.json "$REPO"/results/phase1/full/results/adapters/S2/adapter_model.safetensors "$OUT/adapters/S2/"
cp "$HERE/data/questions_en.jsonl" "$OUT/"
cat > "$OUT/dataset-metadata.json" <<J
{"title": "mmcqa-exp-b-inputs", "id": "muhammadhassan31/mmcqa-exp-b-inputs", "licenses": [{"name": "CC0-1.0"}]}
J
KAGGLE="$REPO/../.venv-local/bin/kaggle"
if "$KAGGLE" datasets status muhammadhassan31/mmcqa-exp-b-inputs >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$OUT" -m "update" --dir-mode zip
else
  "$KAGGLE" datasets create -p "$OUT" --dir-mode zip
fi
