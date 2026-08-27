"""Home construction: tier masks, the non-IID split, and mask application."""
import numpy as np
import pytest

import config as C
from partition import feature_mask, dirichlet_label_partition, assign_tiers, make_homes

FEATS = C.all_features()


def test_feature_mask_matches_the_tier_definition():
    for tier in sorted(C.TIER_FEATURES):
        mask = feature_mask(tier, FEATS)
        observed = {f for f, m in zip(FEATS, mask, strict=True) if m == 1.0}
        assert observed == set(C.features_for_tier(tier))


def test_masks_are_nested_across_tiers():
    m1, m2, m3 = (feature_mask(t, FEATS) for t in (1, 2, 3))
    assert np.all(m1 <= m2) and np.all(m2 <= m3)
    assert m3.sum() == len(FEATS)


def test_dirichlet_partition_is_a_disjoint_cover():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20, dtype=float)
    shards = dirichlet_label_partition(y, n_homes=7, alpha=0.5, rng=rng)
    allidx = np.concatenate(shards)
    assert len(allidx) == len(y)
    assert set(allidx.tolist()) == set(range(len(y)))


def test_dirichlet_partition_produces_label_skew():
    """Non-IID by construction: shard prevalence must not all equal the global rate."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20, dtype=float)
    shards = [s for s in dirichlet_label_partition(y, 10, 0.3, rng) if len(s) > 0]
    rates = [y[s].mean() for s in shards]
    assert np.std(rates) > 0.0


def test_assign_tiers_respects_the_declared_mix():
    rng = np.random.default_rng(0)
    tiers = assign_tiers(20000, {1: 0.5, 2: 0.3, 3: 0.2}, rng)
    for t, want in ((1, 0.5), (2, 0.3), (3, 0.2)):
        assert (tiers == t).mean() == pytest.approx(want, abs=0.02)


def _cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATS)))
    y = (X[:, 0] > 0).astype(float)
    return X, y


def test_make_homes_zeroes_every_unobserved_column():
    X, y = _cohort()
    homes = make_homes(X, y, FEATS, 25, 0.5, {1: 0.5, 2: 0.3, 3: 0.2}, seed=1)
    for h in homes:
        unobserved = h.mask == 0.0
        np.testing.assert_array_equal(h.X[:, unobserved],
                                      np.zeros_like(h.X[:, unobserved]))


def test_make_homes_never_alters_an_observed_measurement():
    """Heterogeneity is imposed by masking only -- observed values pass through."""
    X, y = _cohort()
    homes = make_homes(X, y, FEATS, 25, 0.5, {1: 0.5, 2: 0.3, 3: 0.2}, seed=1)
    seen = 0
    for h in homes:
        obs = h.mask == 1.0
        # Every observed value in the shard must appear unchanged in the cohort.
        assert np.isin(h.X[:, obs], X[:, obs]).all()
        seen += len(h.y)
    assert seen == len(y)  # and the shards partition the cohort exactly


def test_make_homes_drops_empty_shards():
    X, y = _cohort(n=30)
    homes = make_homes(X, y, FEATS, 60, 0.1, {1: 1.0}, seed=3)
    assert all(len(h.y) > 0 for h in homes)
    assert len(homes) <= 60


def test_make_homes_is_deterministic_for_a_fixed_seed():
    X, y = _cohort()
    kw = {"n_homes": 20, "alpha": 0.5, "tier_mix": {1: 0.5, 2: 0.3, 3: 0.2}, "seed": 42}
    a = make_homes(X, y, FEATS, **kw)
    b = make_homes(X, y, FEATS, **kw)
    assert [h.tier for h in a] == [h.tier for h in b]
    for ha, hb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ha.X, hb.X)
        np.testing.assert_array_equal(ha.y, hb.y)
