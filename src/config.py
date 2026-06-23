"""Shared configuration: the device -> feature -> tier mapping.

This is the conceptual core of the study. In the HealthSync Green Box product,
homes buy one of three hardware tiers, so different homes can physically measure
different biometrics. We mirror that exactly: a Tier-k home observes the PROFILE
features plus every feature contributed by a device in tiers 1..k.

Every feature below is a REAL NHANES measured value. Feature *heterogeneity* is
imposed by which devices a simulated home owns (an experimental scenario), never
by altering or fabricating any measurement.
"""
from __future__ import annotations

# Always available: entered once at app setup (no device required).
PROFILE = ["age", "sex_male", "current_smoker"]

# Device -> features it contributes, grouped by the tier that first includes it.
# The smart-pillbox -> medication-count signal is deliberately NOT used: for a
# prevalence label it is diagnosis-adjacent (reverse causation), so including it
# would inflate the top tier artificially. Tier 3's signal is the CGM glycemic
# measurement (HbA1c), which is the physiological quantity a CGM actually reports.
TIER_FEATURES = {
    1: ["bmi", "weight", "sbp", "dbp", "pulse"],  # smart scale + BP cuff
    2: ["waist"],                                  # + InBody body-composition scale
    3: ["hba1c"],                                  # + CGM (glycemic measurement)
}

TIER_NAMES = {1: "Core Essentials", 2: "Advanced Metabolic", 3: "Total Vital"}

# Product premium-credit ladder (for comparison with learned contribution scores).
TIER_CREDIT = {1: 0.10, 2: 0.18, 3: 0.25}

# Target: physician-diagnosed diabetes (NHANES DIQ010 == 1). A genuine condition
# label, not derived from any feature. Lower tiers perform non-invasive risk
# estimation (~0.78 AUROC, matching published risk scores); the CGM tier measures
# the glycemic biomarker directly (~0.92, the clinical gold standard).
LABEL = "diabetes"


def features_for_tier(tier: int) -> list[str]:
    """Features observable by a home at the given tier (cumulative)."""
    feats = list(PROFILE)
    for k in range(1, tier + 1):
        feats += TIER_FEATURES[k]
    return feats


def all_features() -> list[str]:
    """Full feature superset (a Tier-3 home)."""
    return features_for_tier(max(TIER_FEATURES))


# NHANES variable -> our feature name, per raw component file (base name).
# Some features are derived in build_dataset.py (sbp, dbp, current_smoker).
NHANES_VARS = {
    "DEMO": {"SEQN": "SEQN", "RIDAGEYR": "age", "RIAGENDR": "sex_raw"},
    "BMX": {"SEQN": "SEQN", "BMXBMI": "bmi", "BMXWT": "weight", "BMXWAIST": "waist"},
    "BPX": {"SEQN": "SEQN", "BPXSY1": "sy1", "BPXSY2": "sy2", "BPXSY3": "sy3",
            "BPXDI1": "di1", "BPXDI2": "di2", "BPXDI3": "di3", "BPXPLS": "pulse"},
    "GHB": {"SEQN": "SEQN", "LBXGH": "hba1c"},
    "SMQ": {"SEQN": "SEQN", "SMQ020": "smq020", "SMQ040": "smq040"},
    "DIQ": {"SEQN": "SEQN", "DIQ010": "diq010"},
}

MIN_AGE = 20  # adults only
