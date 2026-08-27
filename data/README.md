# Data

This study uses **only real, public** survey data — the U.S. CDC **National Health
and Nutrition Examination Survey (NHANES)**. No synthetic data is generated and no
measurement is perturbed.

* Source: <https://wwwn.cdc.gov/nchs/nhanes/>
* Cycles: **2015–2016** (`_I`) and **2017–2018** (`_J`)
* Access: fully public, no credentialing or DUA required.

`src/download_nhanes.py` fetches the raw SAS-transport (`.xpt`) files into
`data/raw/`; `src/build_dataset.py` merges them on the respondent id `SEQN` into
`data/nhanes_cohort.csv`. Both `data/raw/` and the merged CSV are git-ignored
because they are fully regenerable from the scripts, and because this repository
redistributes no data of its own.

## Verifying you got the same data

`CHECKSUMS.sha256` pins the SHA-256 of every raw file and of the built cohort as
they stood for the published run (downloaded 2026-06-23). From the repository
root:

```bash
sha256sum -c data/CHECKSUMS.sha256
```

A mismatch on a `data/raw/*.xpt` file means CDC has re-released that component
since this study ran; a mismatch on `nhanes_cohort.csv` means the build step
produced a different cohort. Either one rules out an exact-match reproduction —
note which file differs when reporting it.

## Components used

| File   | Variables                              | Role |
|--------|----------------------------------------|------|
| DEMO   | RIDAGEYR, RIAGENDR                      | age, sex (profile) |
| BMX    | BMXBMI, BMXWT, BMXWAIST                 | scale + InBody body composition |
| BPX    | BPXSY1-3, BPXDI1-3, BPXPLS              | BP cuff (systolic/diastolic/pulse) |
| GHB    | LBXGH                                   | HbA1c (CGM/Tier-3 glycemic signal) |
| SMQ    | SMQ040                                  | current-smoker (profile) |
| DIQ    | DIQ010                                  | **label**: physician-diagnosed diabetes |

## Cohort

Adults (age ≥ 20) with a valid diabetes response and **complete** values for every
modeled feature are kept (complete-case; no imputation). Result: **n = 9,360**
(7,488 train / 1,872 test), diabetes prevalence **15.0 %**.

Only complete records are used so that device-tier heterogeneity is imposed purely
by *masking* the features a home's devices cannot measure — never by altering or
fabricating any measured value.
