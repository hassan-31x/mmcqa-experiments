#!/usr/bin/env bash
# Download a finished Kaggle notebook's output and unzip it.
#   local_analysis/pull.sh muhammadhassan31/mmcqa-phase1-finetune phase1 2026-11-05_full
# Result: results/<phase>/<tag>/results/{predictions,adapters,logs,submission,...}
set -euo pipefail
KERNEL="$1"; PHASE="$2"; TAG="$3"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="${KAGGLE_BIN:-$HERE/../.venv-local/bin/kaggle}"
DEST="$HERE/results/$PHASE/$TAG"
mkdir -p "$DEST"
"$KAGGLE" kernels output "$KERNEL" -p "$DEST" --file-pattern ".*\\.zip$"
(cd "$DEST" && unzip -q -o "${PHASE}_results.zip" && ls)
echo "pulled to $DEST"
