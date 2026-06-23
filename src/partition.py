"""Split the cohort into simulated edge nodes ("homes") with device tiers.

Each home holds a non-IID shard of real records (label-skewed via a Dirichlet
draw, the standard FL heterogeneity model) and owns one hardware tier, which
determines the FEATURE MASK applied to its records. Tier assignment is independent
of the data; only device availability differs, never the measured values.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config as C


@dataclass
class Home:
    idx: int
    tier: int
    X: np.ndarray          # (n, D) standardized, unobserved cols zeroed
    y: np.ndarray          # (n,)
    mask: np.ndarray       # (D,) 1 where the tier's devices observe the feature


def feature_mask(tier: int, feat_order: list[str]) -> np.ndarray:
    obs = set(C.features_for_tier(tier))
    return np.array([1.0 if f in obs else 0.0 for f in feat_order])


def dirichlet_label_partition(y: np.ndarray, n_homes: int, alpha: float,
                              rng: np.random.Generator) -> list[np.ndarray]:
    """Assign sample indices to homes with a per-class Dirichlet split (non-IID)."""
    idx_by_class = {c: rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    shards: list[list[int]] = [[] for _ in range(n_homes)]
    for c, idx in idx_by_class.items():
        props = rng.dirichlet([alpha] * n_homes)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for h, part in enumerate(np.split(idx, cuts)):
            shards[h].extend(part.tolist())
    return [np.array(sorted(s)) for s in shards]


def assign_tiers(n_homes: int, mix: dict[int, float],
                 rng: np.random.Generator) -> np.ndarray:
    tiers = sorted(mix)
    probs = np.array([mix[t] for t in tiers], dtype=float)
    probs /= probs.sum()
    return rng.choice(tiers, size=n_homes, p=probs)


def make_homes(Xtr: np.ndarray, ytr: np.ndarray, feat_order: list[str],
               n_homes: int, alpha: float, tier_mix: dict[int, float],
               seed: int) -> list[Home]:
    rng = np.random.default_rng(seed)
    shards = dirichlet_label_partition(ytr, n_homes, alpha, rng)
    tiers = assign_tiers(n_homes, tier_mix, rng)
    homes = []
    for i, (idx, tier) in enumerate(zip(shards, tiers)):
        if len(idx) == 0:
            continue
        mask = feature_mask(int(tier), feat_order)
        Xi = Xtr[idx] * mask  # zero unobserved feature columns (device absent)
        homes.append(Home(i, int(tier), Xi, ytr[idx], mask))
    return homes
