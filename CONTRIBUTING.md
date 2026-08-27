# Contributing

This repository is a **research artifact**: it exists so that the results in the
accompanying AIBThings 2026 paper can be re-run and checked. That goal shapes what
kinds of changes are welcome.

## Ground rules

1. **The published numbers are frozen.** `results/results.json`, `results/comparison.csv`
   and `figures/*.png` are the exact artifacts reported in the camera-ready paper.
   Do not overwrite them. If a change alters any reported quantity, say so
   explicitly in the pull-request description and leave the committed artifacts
   untouched — a follow-up run belongs in a new file, not on top of the old one.
2. **Real data only.** Every modeled value comes from NHANES. No synthetic records,
   no imputation, no perturbation of a measurement. Device-tier heterogeneity is
   imposed by *masking* features a home's devices cannot observe, never by editing
   data. Test fixtures may use hand-built arrays, but no test result may feed a
   reported number.
3. **No LLMs and no GPU in the pipeline.** The study's premise is that this runs on
   a consumer CPU.
4. **Estimated energy is never reported as measured.** Anything derived from the TDP
   fallback must stay tagged `energy_source: "tdp_estimate"`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Before opening a pull request

```bash
ruff check src scripts tests     # lint
python -m pytest                 # unit tests + README/results consistency
python scripts/verify_results.py # published results self-check
```

If you touched anything in `src/`, also confirm the full run still reproduces:

```bash
cp results/results.json /tmp/pinned.json
python src/run_experiment.py
python scripts/verify_results.py --reference /tmp/pinned.json
git checkout -- results/ figures/       # restore the published artifacts
```

`verify_results.py` compares every reported quantity except energy, which is
hardware-dependent by nature.

## What is especially welcome

* Ports of the tier-aware aggregation rule to other datasets with genuine feature
  heterogeneity.
* Additional baselines (feature-imputation FL, per-tier personalization).
* Independent reproductions on other CPUs, reported as an issue with the
  `verify_results.py` output attached.

## What is out of scope

* Restructuring the code into a package or adding a framework dependency. The flat
  `src/` layout is intentional: each module maps to one section of the paper.
* Swapping the linear model for a deep one. The model is linear so that a weight
  can be masked to exactly one device.

## Reporting a problem with the results

Open an issue with the output of `python scripts/verify_results.py`, your OS and
CPU, `python --version`, and `pip freeze`. A mismatch in `data/CHECKSUMS.sha256`
usually means CDC has re-released an NHANES component since this study ran; note
which file differs.
