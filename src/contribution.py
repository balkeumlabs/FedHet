"""Tier contribution scoring -- the FLAI / sPEG incentive tie-in (analytical).

We quantify how much predictive signal each device tier adds, then turn those
marginal gains into a normalized reward ladder. This connects the learning result
to the product's premium-credit ladder (Tier 1/2/3 -> 10/18/25%): homes that own
more devices contribute more signal and should be rewarded more. No blockchain run
is required; this mirrors the on-chain reward logic specified in the FLAI Protocol.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import config as C
from federated import train_centralized


def _auc_with_features(Xtr, ytr, Xte, yte, feat_order, feats, epochs, lr, l2):
    cols = [i for i, f in enumerate(feat_order) if f in feats]
    m = train_centralized(Xtr[:, cols], ytr, epochs=epochs, lr=lr, l2=l2)
    return roc_auc_score(yte, m.proba(Xte[:, cols]))


def tier_contributions(Xtr, ytr, Xte, yte, feat_order, *, epochs, lr, l2):
    """Marginal AUROC gain as each tier's devices are added on top of PROFILE."""
    baseline_feats = set(C.PROFILE)
    base_auc = _auc_with_features(Xtr, ytr, Xte, yte, feat_order,
                                  baseline_feats, epochs, lr, l2)
    rows = []
    prev_auc = base_auc
    cum = set(C.PROFILE)
    for t in sorted(C.TIER_FEATURES):
        cum = cum | set(C.TIER_FEATURES[t])
        auc = _auc_with_features(Xtr, ytr, Xte, yte, feat_order, cum, epochs, lr, l2)
        rows.append({"tier": t, "name": C.TIER_NAMES[t],
                     "cum_auc": auc, "marginal_auc_gain": auc - prev_auc,
                     "product_credit": C.TIER_CREDIT[t]})
        prev_auc = auc

    gains = np.array([max(r["marginal_auc_gain"], 0.0) for r in rows])
    # cumulative value of owning up to tier t (sum of marginal gains)
    cumval = np.cumsum(gains)
    reward = cumval / cumval[-1] if cumval[-1] > 0 else np.ones(len(rows))
    for r, rv in zip(rows, reward):
        r["learned_reward_share"] = float(rv)
    return {"profile_auc": base_auc, "tiers": rows}
