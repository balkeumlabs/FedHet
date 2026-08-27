"""Aggregation rules -- the paper's contribution lives in `_aggregate`."""
import numpy as np
import pytest

from models import LinearModel, init_model
from federated import _aggregate, federated_scaler, apply_scaler, train_federated
from partition import Home


def _upd(w, n, mask):
    return (LinearModel(np.asarray(w, dtype=float), 0.0), n, np.asarray(mask, float))


def test_plain_fedavg_is_the_sample_weighted_mean():
    updates = [_upd([2.0, 4.0], 10, [1, 1]), _upd([0.0, 0.0], 30, [1, 1])]
    out = _aggregate(init_model(2), updates, restrict=None, per_feature=False)
    np.testing.assert_allclose(out.w, [0.5, 1.0])  # (10*2 + 30*0) / 40


def test_naive_union_dilutes_a_feature_only_some_homes_observe():
    """The failure mode the paper names: non-observers drag the weight toward init."""
    observing = _upd([0.0, 8.0], 25, [1, 1])
    non_observing = _upd([0.0, 0.0], 75, [1, 0])  # weight stayed at its init
    out = _aggregate(init_model(2), [observing, non_observing],
                     restrict=None, per_feature=False)
    # Averaged over ALL 100 samples: 25*8/100 = 2.0, a quarter of the true value.
    assert out.w[1] == pytest.approx(2.0)


def test_tier_aware_recovers_the_undiluted_weight():
    """Same updates, per-feature denominator -> the observers' value survives intact."""
    observing = _upd([0.0, 8.0], 25, [1, 1])
    non_observing = _upd([0.0, 0.0], 75, [1, 0])
    out = _aggregate(init_model(2), [observing, non_observing],
                     restrict=None, per_feature=True)
    assert out.w[1] == pytest.approx(8.0)  # averaged over the 25 observing samples only


def test_tier_aware_and_plain_agree_when_every_home_observes_everything():
    updates = [_upd([1.0, 3.0], 10, [1, 1]), _upd([5.0, 7.0], 30, [1, 1])]
    a = _aggregate(init_model(2), updates, None, per_feature=False)
    b = _aggregate(init_model(2), updates, None, per_feature=True)
    np.testing.assert_allclose(a.w, b.w)


def test_restrict_mask_zeroes_out_excluded_features():
    updates = [_upd([2.0, 9.0], 10, [1, 1])]
    prev = init_model(2)
    out = _aggregate(prev, updates, restrict=np.array([1.0, 0.0]), per_feature=False)
    assert out.w[0] == pytest.approx(2.0)
    assert out.w[1] == prev.w[1]  # never aggregated -> keeps its previous value


def test_unobserved_feature_keeps_its_previous_global_value():
    prev = LinearModel(np.array([0.0, 1.234]), 0.0)
    updates = [_upd([2.0, 0.0], 10, [1, 0])]
    out = _aggregate(prev, updates, restrict=None, per_feature=True)
    assert out.w[1] == pytest.approx(1.234)


def test_bias_is_always_averaged_over_all_homes():
    u1 = (LinearModel(np.zeros(2), 1.0), 10, np.array([1.0, 0.0]))
    u2 = (LinearModel(np.zeros(2), 3.0), 30, np.array([1.0, 1.0]))
    out = _aggregate(init_model(2), [u1, u2], None, per_feature=True)
    assert out.b == pytest.approx((10 * 1.0 + 30 * 3.0) / 40)


# ----- federated standardization -------------------------------------------
def _homes_for_scaler():
    # Home A observes both features; home B observes only feature 0.
    a = Home(0, 3, np.array([[1.0, 10.0], [3.0, 20.0]]), np.array([0.0, 1.0]),
             np.array([1.0, 1.0]))
    b = Home(1, 1, np.array([[5.0, 0.0], [7.0, 0.0]]), np.array([1.0, 0.0]),
             np.array([1.0, 0.0]))
    return [a, b]


def test_federated_scaler_matches_the_pooled_statistics_of_observing_homes():
    mean, std = federated_scaler(_homes_for_scaler(), ["f0", "f1"])
    np.testing.assert_allclose(mean, [np.mean([1, 3, 5, 7]), np.mean([10, 20])])
    np.testing.assert_allclose(std, [np.std([1, 3, 5, 7]), np.std([10, 20])])


def test_scaler_ignores_the_zeros_of_homes_that_cannot_observe():
    """Home B's zeros in column 1 are 'device absent', not a measurement of 0."""
    mean, _ = federated_scaler(_homes_for_scaler(), ["f0", "f1"])
    assert mean[1] == pytest.approx(15.0)  # not (10+20+0+0)/4 = 7.5


def test_apply_scaler_leaves_unobserved_columns_exactly_zero():
    mean, std = federated_scaler(_homes_for_scaler(), ["f0", "f1"])
    h = _homes_for_scaler()[1]
    Xs = apply_scaler(h.X, h.mask, mean, std)
    np.testing.assert_array_equal(Xs[:, 1], np.zeros(len(Xs)))
    np.testing.assert_allclose(Xs[:, 0], (h.X[:, 0] - mean[0]) / std[0])


# ----- driver ---------------------------------------------------------------
def _small_homes(seed=0):
    rng = np.random.default_rng(seed)
    homes = []
    for i, tier in enumerate([1, 1, 3, 3]):
        X = rng.normal(size=(30, 10))
        mask = np.ones(10)
        if tier == 1:
            mask[8:] = 0.0
        homes.append(Home(i, tier, X * mask, (X[:, 0] > 0).astype(float), mask))
    return homes


FEATS = ["age", "sex_male", "current_smoker", "bmi", "weight", "sbp", "dbp",
         "pulse", "waist", "hba1c"]


@pytest.mark.parametrize("method",
                         ["intersection_fl", "naive_union_fl", "tieraware_fl"])
def test_training_is_deterministic_for_a_fixed_seed(method):
    kw = {"method": method, "rounds": 3, "local_epochs": 2, "lr": 0.5,
          "l2": 1e-4, "clients_per_round": 2, "seed": 7}
    m1, _, h1 = train_federated(_small_homes(), FEATS, **kw)
    m2, _, h2 = train_federated(_small_homes(), FEATS, **kw)
    np.testing.assert_array_equal(m1.w, m2.w)
    assert h1 == h2


def test_intersection_fl_never_learns_a_higher_tier_weight():
    model, restrict, _ = train_federated(
        _small_homes(), FEATS, method="intersection_fl", rounds=5, local_epochs=3,
        lr=0.5, l2=1e-4, clients_per_round=None, seed=7)
    # waist (Tier 2) and hba1c (Tier 3) are outside the intersection.
    np.testing.assert_array_equal(model.w[8:], np.zeros(2))
    np.testing.assert_array_equal(restrict[8:], np.zeros(2))


def test_tier_aware_fl_does_learn_the_top_tier_weight():
    model, restrict, _ = train_federated(
        _small_homes(), FEATS, method="tieraware_fl", rounds=5, local_epochs=3,
        lr=0.5, l2=1e-4, clients_per_round=None, seed=7)
    assert restrict is None
    assert np.any(model.w[8:] != 0.0)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        train_federated(_small_homes(), FEATS, method="fedavg_typo", rounds=1,
                        local_epochs=1, lr=0.5, l2=0.0, clients_per_round=None,
                        seed=0)


def test_history_has_one_entry_per_round():
    _, _, hist = train_federated(_small_homes(), FEATS, method="tieraware_fl",
                                 rounds=4, local_epochs=1, lr=0.5, l2=0.0,
                                 clients_per_round=None, seed=0,
                                 eval_fn=lambda m, r: float(m.b))
    assert len(hist) == 4
