#!/usr/bin/env bash
# Reproduce the full study end-to-end on a CPU-only consumer machine.
#
# Usage:
#   bash scripts/reproduce.sh              # download -> build -> run -> verify
#   SKIP_VERIFY=1 bash scripts/reproduce.sh   # skip the comparison step
#
# The run overwrites results/ and figures/ in your working tree. Those files are
# the published artifacts, so `git checkout -- results/ figures/` restores them.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo ">> creating virtualenv"
  "$PY" -m venv .venv
fi
VENV_PY=".venv/bin/python"

echo ">> installing dependencies"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt

# Energy is read from the Linux powercap RAPL interface, which is root-only on most
# current distributions. Without it the harness still runs, but every energy figure
# is a labeled TDP estimate rather than a measurement.
if ! "$VENV_PY" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from measure import rapl_status
ok, reason = rapl_status()
print(f">> energy: {'MEASURED via RAPL' if ok else 'NOT measurable'} -- {reason}")
sys.exit(0 if ok else 1)
PYEOF
then
  echo ">>"
  echo ">> Energy will be ESTIMATED from TDP, not measured. To reproduce the"
  echo ">> measured energy figures, grant read access and re-run:"
  echo ">>     sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj"
  echo ">>"
fi

echo ">> [1/4] downloading raw NHANES files from CDC"
"$VENV_PY" src/download_nhanes.py

echo ">> [2/4] building the analysis cohort (real, complete-case records)"
"$VENV_PY" src/build_dataset.py

echo ">> [3/4] verifying data checksums against the published run"
if sha256sum -c data/CHECKSUMS.sha256; then
  echo ">> data matches the published run exactly"
else
  echo ">> WARNING: data differs from the published run (CDC may have re-released a"
  echo ">>          component). Results below will not match exactly."
fi

# The reproduction writes into runs/local/ so that the published artifacts in
# results/ and figures/ stay untouched and can serve as the comparison reference.
OUT_DIR=${OUT_DIR:-runs/local}
echo ">> [4/4] running the federated experiment (writes $OUT_DIR/)"
"$VENV_PY" src/run_experiment.py --out-dir "$OUT_DIR" --fig-dir "$OUT_DIR/figures"

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">> comparing $OUT_DIR/results.json against the published results/results.json"
  "$VENV_PY" scripts/verify_results.py --candidate "$OUT_DIR/results.json" || {
    echo ">> Some quantities did not match. See the FAILED lines above."
    exit 1
  }
fi

echo ">> done. Your run: $OUT_DIR/results.json, comparison.csv, figures/*.png"
echo ">> Published reference (unchanged): results/results.json, figures/*.png"
