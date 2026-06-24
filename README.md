# FedHet — Federated Diabetes-Risk Scoring under Device-Tier Heterogeneity

Reproducible research artifact for a Balkeum Labs paper submitted to **AIBThings
2026** (IEEE Int. Conf. on AI, Blockchain, and IoT). CPU-only, no GPU, no LLM, and
**real public data only** (NHANES) — no synthetic data, no differential privacy.

> **Problem.** In a real edge-health deployment (the HealthSync Green Box), homes
> buy different *hardware tiers*, so different homes can physically measure
> different biometrics. Standard federated averaging assumes every client shares
> the same feature space — which this *device-tier feature heterogeneity* breaks.
>
> **Result.** Handling heterogeneity naively (training only on the features
> **every** home has) throws away **15.6 AUROC points**. A simple **tier-aware**
> aggregation — averaging each weight only over the homes whose devices observe
> that feature — recovers **+13.9 points**, lands **within 1.6 points of the
> centralized upper bound**, and converges in **2 rounds instead of 26** — even
> though only ~25% of homes own the top-tier (CGM) device. All measured on a
> consumer CPU: **~0.3 s**, **~6.6 J** of real package energy, **44 bytes** uplink
> per home per round, and **zero raw-data egress**.

## TL;DR results

| Method | Test AUROC | AUPRC | Gap vs centralized | Rounds to converge |
|---|---|---|---|---|
| Centralized (upper bound) | **0.927** | 0.733 | — | — |
| Intersection FL (Tier-1 only) | 0.771 | 0.330 | −15.6 pp | 3 (stuck) |
| Naive union FL | 0.904 | 0.675 | −2.2 pp | 26 |
| **Tier-aware FL (ours)** | **0.910** | 0.692 | **−1.6 pp** | **2** |

Per-tier marginal contribution (drives a fair, FLAI-style reward ladder):
Tier 1 **+5.0 pp**, Tier 2 **+1.0 pp**, Tier 3 (CGM) **+14.6 pp** AUROC — monotone,
and tracking the product's 10 % / 18 % / 25 % premium-credit ladder.

![AUROC](figures/fig1_auroc.png)
![Convergence](figures/fig2_convergence.png)

## The setup

* **Data:** NHANES 2015–2016 + 2017–2018, adults (age ≥ 20), complete-case
  (n ≈ 9,360; diabetes prevalence ≈ 15 %). See [`data/README.md`](data/README.md).
* **Label:** physician-diagnosed diabetes (`DIQ010`). A genuine condition label,
  not derived from any feature.
* **Devices → features → tiers** (mirrors the product hardware tiers):

  | Tier | Device added | Features | Cumulative AUROC |
  |---|---|---|---|
  | *profile* | (app setup) | age, sex, smoker | 0.72 |
  | **T1** Core | smart scale + BP cuff | + BMI, weight, systolic, diastolic, pulse | 0.77 |
  | **T2** Advanced | + InBody body-composition scale | + waist | 0.78 |
  | **T3** Total Vital | + CGM | + HbA1c | 0.93 |

  Lower tiers do *non-invasive* risk estimation (~0.78 AUROC, in line with
  published scores such as FINDRISC/ADA); the CGM tier measures the glycemic
  biomarker directly (~0.92, the clinical gold standard).

* **Model:** plain logistic regression (NumPy), trained with FedAvg. A linear
  model is deliberate — it federates cleanly, trains in milliseconds on a CPU, and
  lets us mask features exactly to the device a home owns.
* **Privacy substrate:** only model weights / aggregate statistics ever leave a
  home; under secure multi-party computation (SMPC) the server sees only the
  masked sum. SMPC is the assumed privacy layer, not the focus of this study.
* **Incentive tie-in:** per-tier marginal value is turned into a normalized reward
  ladder, mirroring the on-chain reward logic of Balkeum's FLAI Protocol. This is
  computed analytically — no blockchain deployment is required.

### Methods compared

* **Centralized** — pool all data + all features (relative upper bound).
* **Intersection FL** — standard FedAvg on the feature set *every* home has
  (profile + Tier 1); richer-tier signals are discarded.
* **Naive union FL** — full feature superset, but plain FedAvg averages every
  weight over *all* homes, so homes lacking a feature dilute it.
* **Tier-aware FL (ours)** — full superset with **per-feature, participation-
  weighted** aggregation: each weight is averaged only over the homes whose
  devices observe that feature.

## Reproduce

CPU-only; developed on an AMD Ryzen 5 5500GT (6C/12T), 62 GB RAM, no GPU used.

```bash
bash scripts/reproduce.sh
```

or step by step:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_nhanes.py     # fetch raw NHANES .xpt from CDC
python src/build_dataset.py       # -> data/nhanes_cohort.csv
python src/run_experiment.py      # -> results/*.json,*.csv and figures/*.png
```

Outputs are deterministic (fixed seed). Energy is read from the Linux powercap
RAPL interface when available, otherwise a labeled TDP estimate is used.

## Repository layout

```
src/
  config.py          device -> feature -> tier mapping; label definition
  download_nhanes.py  fetch raw NHANES files from the CDC public server
  build_dataset.py    merge components on SEQN -> complete-case cohort
  partition.py        split into homes, assign tiers, apply feature masks
  models.py           NumPy logistic regression with per-feature gradient masking
  federated.py        federated scaler + the 3 aggregation strategies
  contribution.py     per-tier marginal contribution / reward ladder
  measure.py          RAPL energy, CPU time, communication accounting
  run_experiment.py   orchestrates everything; writes results + figures
results/              results.json, comparison.csv
figures/              fig1..fig4 (.png)
data/                 README (raw data are downloaded, not committed)
```

## Honesty notes & limitations

* **Real data only.** Every value is a measured NHANES value; tier heterogeneity
  is imposed by *masking* devices a home lacks, never by altering data.
* **Why diabetes, not a "polypharmacy" feature.** An earlier design used
  medication count as a "smart-pillbox" proxy; we dropped it because for a
  prevalence label it is diagnosis-adjacent (reverse causation) and inflates the
  top tier artificially. Tier 3's signal is the CGM glycemic measurement.
* **Cross-sectional.** NHANES is cross-sectional; this is risk *stratification*,
  not incident-risk prediction. Homes are simulated by partitioning real records.
* **Blockchain is analytical.** The reward ladder mirrors the FLAI Protocol's
  on-chain settlement logic; no chain is deployed here.
* **Not a medical device.** Research artifact only; no clinical claims.

## Authors

- **Eli Yune** ([@yune1ha](https://github.com/yune1ha)) — Balkeum Labs
- **Umer Majeed** ([@umerblabs](https://github.com/umerblabs)) — Balkeum Labs

## License

Code: [MIT](LICENSE), © 2026 Balkeum Labs. NHANES data are U.S. public-domain.
