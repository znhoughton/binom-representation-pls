#!/usr/bin/env bash
# Pythia supplementary: by-layer training-dynamics at BabyLM-comparable token counts
#
# Runs embedding extraction + MLP + corpus-freq regression at the six checkpoint
# steps whose token counts match BabyLM's log-spaced intermediate checkpoints:
#
#   step16   ≈  33.6M tokens   (≈ BabyLM ~0.6% training)
#   step32   ≈  67.1M tokens   (≈ BabyLM ~1.2%)
#   step64   ≈  134M  tokens   (≈ BabyLM ~3.6%)
#   step256  ≈  537M  tokens   (≈ BabyLM ~9–10%)
#   step512  ≈  1.07B tokens   (≈ BabyLM ~23%)
#   step1000 ≈  2.10B tokens   (≈ BabyLM full training boundary)
#
# After the pipeline finishes, runs corpus_freq_regression_checkpoints_pythia.R
# for each completed model to produce the by-layer β CSV.
#
# Usage:
#   bash Scripts/run_pythia_supplementary_babylm_checkpoints.sh
#   bash Scripts/run_pythia_supplementary_babylm_checkpoints.sh --models 160m 1b
#   bash Scripts/run_pythia_supplementary_babylm_checkpoints.sh --skip-controls

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(160m 410m 1b 2.8b)
GPU=0
EXTRA_ARGS=()

# Parse --models override; pass everything else through to run_pythia_checkpoints.py
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1"); shift
      done
      ;;
    --gpu)
      GPU="$2"; shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1"); shift
      ;;
  esac
done

echo "======================================================================"
echo "  Pythia supplementary: BabyLM-comparable checkpoint analysis"
echo "  Models : ${MODELS[*]}"
echo "  Steps  : step16, step32, step64, step256, step512, step1000"
echo "  Tokens :  33.6M  67.1M  134M   537M    1.07B   2.10B"
echo "======================================================================"
echo ""

python "$SCRIPT_DIR/run_pythia_checkpoints.py" \
  --models "${MODELS[@]}" \
  --range babylm \
  --gpu "$GPU" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

PIPELINE_EXIT=$?

echo ""
echo "======================================================================"
if [[ $PIPELINE_EXIT -eq 0 ]]; then
  echo "  DONE"
else
  echo "  DONE (pipeline exited with code $PIPELINE_EXIT — check logs above)"
fi
echo "======================================================================"

exit $PIPELINE_EXIT
