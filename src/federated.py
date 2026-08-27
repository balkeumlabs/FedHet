"""Federated training: the global scaler and the three aggregation strategies.

Methods compared
  centralized      : pool all data + all features (relative upper bound).
  intersection_fl  : standard FedAvg restricted to the feature set EVERY home has
                     (PROFILE + Tier-1) -- the lowest common denominator.
  naive_union_fl   : full feature superset, but plain FedAvg that averages every
                     weight over ALL homes (homes lacking a feature dilute it).
  tieraware_fl     : full superset with PER-FEATURE participation-weighted
                     aggregation -- each weight is averaged only over the homes
                     whose devices observe that feature.  (ours)

Only model weights and aggregate statistics ever leave a home; under SMPC the
server sees only the masked sum (privacy layer, cited but not the focus here).
"""
from __future__ import annotations

import numpy as np

import config as C
from models import LinearModel, init_model, local_train
from partition import Home


# ----- federated standardization (SMPC-compatible: only count/sum/sumsq leave) --
def federated_scaler(homes: list[Home], feat_order: list[str]):
    """Global per-feature mean/std aggregated across homes that observe it."""
    D = len(feat_order)
    s = np.zeros(D); ss = np.zeros(D); cnt = np.zeros(D)
    for h in homes:
        obs = h.mask.astype(bool)
        Xo = h.X[:, obs]
        s[obs] += Xo.sum(axis=0)
        ss[obs] += (Xo ** 2).sum(axis=0)
        cnt[obs] += len(h.y)
    cnt = np.maximum(cnt, 1)
    mean = s / cnt
    var = np.maximum(ss / cnt - mean ** 2, 1e-8)
    return mean, np.sqrt(var)


def apply_scaler(X: np.ndarray, mask: np.ndarray, mean, std) -> np.ndarray:
    Xs = (X - mask * mean) / std
    return Xs * mask  # keep unobserved columns at 0


# ----- aggregation ----------------------------------------------------------
def _aggregate(global_model: LinearModel, updates, restrict: np.ndarray | None,
               per_feature: bool) -> LinearModel:
    """Sample-weighted FedAvg. ``updates`` is a list of ``(model, n, mask)``.

    ``per_feature=False`` reproduces plain FedAvg: every weight is averaged over
    ALL participating homes, so homes that cannot observe a feature (and therefore
    hold that weight at its initial value) still dilute it.

    ``per_feature=True`` is the tier-aware rule: weight *j* is averaged only over
    the homes whose devices actually observe feature *j*, i.e. the denominator is
    the participating sample count for that feature rather than the global one.

    The bias has no feature, so it is always averaged over all homes. A feature
    that no participating home observes keeps its previous global value.
    """
    D = len(global_model.w)
    w_num = np.zeros(D); w_den = np.zeros(D)
    b_num = 0.0; b_den = 0.0
    for m, n, mask in updates:
        weight = mask if per_feature else np.ones(D)
        if restrict is not None:
            weight = weight * restrict
        w_num += n * weight * m.w
        w_den += n * weight
        b_num += n * m.b
        b_den += n
    observed = w_den > 0
    new_w = np.where(observed, w_num / np.where(observed, w_den, 1.0),
                     global_model.w)
    return LinearModel(new_w, b_num / max(b_den, 1.0))


# ----- training driver ------------------------------------------------------
def train_federated(homes: list[Home], feat_order: list[str], *, method: str,
                    rounds: int, local_epochs: int, lr: float, l2: float,
                    clients_per_round: int | None, seed: int,
                    eval_fn=None):
    """Run a federated method. Returns (model, restrict_mask, history).

    eval_fn(model, restrict) -> float is called each round for a convergence curve.
    """
    rng = np.random.default_rng(seed)
    D = len(feat_order)
    model = init_model(D)

    if method == "intersection_fl":
        obs = set(C.features_for_tier(1))                 # PROFILE + Tier 1
        restrict = np.array([1.0 if f in obs else 0.0 for f in feat_order])
        per_feature = False
    elif method == "naive_union_fl":
        restrict = None
        per_feature = False
    elif method == "tieraware_fl":
        restrict = None
        per_feature = True
    else:
        raise ValueError(method)

    history = []
    for _round in range(rounds):
        if clients_per_round and clients_per_round < len(homes):
            sel = rng.choice(len(homes), clients_per_round, replace=False)
            cohort = [homes[i] for i in sel]
        else:
            cohort = homes
        updates = []
        for h in cohort:
            cmask = h.mask * restrict if restrict is not None else h.mask
            local = local_train(model, h.X, h.y, cmask, local_epochs, lr, l2)
            updates.append((local, len(h.y), cmask))
        model = _aggregate(model, updates, restrict, per_feature)
        if eval_fn is not None:
            history.append(eval_fn(model, restrict))
    return model, restrict, history


def train_centralized(X: np.ndarray, y: np.ndarray, *, epochs: int, lr: float,
                      l2: float) -> LinearModel:
    model = init_model(X.shape[1])
    full = np.ones(X.shape[1])
    return local_train(model, X, y, full, epochs, lr, l2)
