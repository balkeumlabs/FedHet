"""Merge raw NHANES components into one analysis table of real, complete records.

Output: data/nhanes_cohort.csv with columns = PROFILE + all tier features + label.

Honesty notes:
  * Every value is a real measured NHANES value.
  * We keep only records that are COMPLETE across every feature and the label, so
    no imputation/fabrication is ever needed. Device heterogeneity is later imposed
    by masking (partition.py), not by altering data.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import config as C

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "nhanes_cohort.csv"
CYCLE_SUFFIX = {"2015-2016": "_I", "2017-2018": "_J"}


def read_xpt(path: Path, keep: list[str]) -> pd.DataFrame:
    df, _ = pyreadstat.read_xport(str(path))
    have = [c for c in keep if c in df.columns]
    return df[have].copy()


def build_cycle(suf: str) -> pd.DataFrame:
    def load(comp: str) -> pd.DataFrame:
        return read_xpt(RAW / f"{comp}{suf}.xpt", list(C.NHANES_VARS[comp]))

    demo = load("DEMO").rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex_raw"})
    bmx = load("BMX").rename(columns={"BMXBMI": "bmi", "BMXWT": "weight",
                                      "BMXWAIST": "waist"})
    bpx = load("BPX")
    ghb = load("GHB").rename(columns={"LBXGH": "hba1c"})
    smq = load("SMQ")
    diq = load("DIQ")

    df = demo[["SEQN", "age", "sex_raw"]].copy()
    df = df.merge(bmx[["SEQN", "bmi", "weight", "waist"]], on="SEQN", how="left")

    # Blood pressure: average the valid (non-zero) replicate readings.
    sy = bpx[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan)
    di = bpx[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan)
    bp = pd.DataFrame({"SEQN": bpx["SEQN"],
                       "sbp": sy.mean(axis=1, skipna=True),
                       "dbp": di.mean(axis=1, skipna=True),
                       "pulse": bpx["BPXPLS"]})
    df = df.merge(bp, on="SEQN", how="left")
    df = df.merge(ghb[["SEQN", "hba1c"]], on="SEQN", how="left")

    # Sex: male = 1.
    df["sex_male"] = (df["sex_raw"] == 1).astype(float)

    # Current smoker: SMQ040 in {1 (every day), 2 (some days)}; everyone else 0.
    smq = smq.rename(columns={"SMQ040": "smq040"})
    df = df.merge(smq[["SEQN", "smq040"]], on="SEQN", how="left")
    df["current_smoker"] = df["smq040"].isin([1, 2]).astype(float)

    # Label: physician-diagnosed diabetes (DIQ010 == 1 "yes"). Codes 1/2/3
    # (yes / no / borderline) are valid responses; borderline is treated as
    # negative. Other codes (refused/don't know/missing) -> label unknown.
    diq = diq.rename(columns={"DIQ010": "diq010"})
    diabetes = (diq["diq010"] == 1).astype(float)
    valid = diq["diq010"].isin([1, 2, 3])
    lab = pd.DataFrame({"SEQN": diq["SEQN"], "diabetes": diabetes,
                        "_lab_valid": valid})
    df = df.merge(lab, on="SEQN", how="left")
    return df


def main() -> int:
    frames = []
    for cyc, suf in CYCLE_SUFFIX.items():
        f = build_cycle(suf)
        f["cycle"] = cyc
        frames.append(f)
        print(f"[{cyc}] rows={len(f)}")
    df = pd.concat(frames, ignore_index=True)

    df = df[df["age"] >= C.MIN_AGE]
    df = df[df["_lab_valid"] == True]  # noqa: E712

    cols = C.all_features() + [C.LABEL]
    before = len(df)
    df = df.dropna(subset=cols)  # keep only fully real, complete records
    print(f"adults w/ valid label: {before}; complete-case: {len(df)}")

    out = df[["cycle"] + cols].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    pos = int(out[C.LABEL].sum())
    print(f"\nSaved {OUT}  (n={len(out)}, features={len(C.all_features())})")
    print(f"{C.LABEL} prevalence: {pos}/{len(out)} = {pos/len(out)*100:.1f}%")
    print("feature means:\n", out[C.all_features()].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
