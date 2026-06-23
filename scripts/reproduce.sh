#!/usr/bin/env bash
# Reproduce the full study end-to-end on a CPU-only consumer machine.
# Usage:  bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
if [ ! -d .venv ]; then
  echo ">> creating virtualenv"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ">> [1/3] downloading raw NHANES files from CDC"
python src/download_nhanes.py

echo ">> [2/3] building the analysis cohort (real, complete-case records)"
python src/build_dataset.py

echo ">> [3/3] running the federated experiment (writes results/ and figures/)"
python src/run_experiment.py

echo ">> done. See results/results.json, results/comparison.csv, figures/*.png"
