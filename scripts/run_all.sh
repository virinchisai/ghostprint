#!/usr/bin/env bash
# End-to-end reproduction. Assumes venv active and data/LibriSpeech present
# (see scripts/download_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_data.py
python scripts/extract_features.py all
python scripts/train.py
python scripts/evaluate.py
python scripts/ivr_experiment.py
echo "All done — see results/"
