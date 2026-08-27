"""Logistic-regression primitives, especially the per-feature gradient masking."""
import numpy as np
import pytest

from models import LinearModel, init_model, local_train, _sigmoid


def _toy(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    return X, y


def test_sigmoid_is_stable_at_extreme_logits():
    z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        p = _sigmoid(z)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[2] == pytest.approx(0.5)
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[4] == pytest.approx(1.0, abs=1e-12)


def test_masked_weights_never_move():
    """A home must not update a weight its devices cannot inform."""
    X, y = _toy()
    mask = np.array([1.0, 1.0, 0.0, 0.0])
    start = init_model(X.shape[1])
    start.w = np.array([0.3, -0.2, 7.0, -7.0])  # arbitrary nonzero masked weights
    out = local_train(start, X, y, mask, epochs=50, lr=0.5, l2=1e-4)
    assert out.w[2] == 7.0 and out.w[3] == -7.0
    assert not np.allclose(out.w[:2], start.w[:2])


def test_local_train_decreases_logistic_loss():
    X, y = _toy()
    mask = np.ones(X.shape[1])
    m0 = init_model(X.shape[1])
    m1 = local_train(m0, X, y, mask, epochs=200, lr=0.5, l2=0.0)

    def nll(m):
        p = np.clip(m.proba(X), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    assert nll(m1) < nll(m0)


def test_local_train_does_not_mutate_the_input_model():
    """Clients receive the global model by value; mutating it would corrupt the round."""
    X, y = _toy()
    m0 = init_model(X.shape[1])
    m0.w = np.array([0.1, 0.2, 0.3, 0.4])
    before = m0.w.copy()
    local_train(m0, X, y, np.ones(4), epochs=10, lr=0.5, l2=0.0)
    np.testing.assert_array_equal(m0.w, before)


def test_bias_is_unmasked():
    """The bias belongs to no feature, so every home updates it."""
    X, y = _toy()
    out = local_train(init_model(4), X, y, np.zeros(4), epochs=20, lr=0.5, l2=0.0)
    np.testing.assert_array_equal(out.w, np.zeros(4))
    assert out.b != 0.0


def test_copy_is_deep():
    m = LinearModel(np.array([1.0, 2.0]), 0.5)
    c = m.copy()
    c.w[0] = 99.0
    assert m.w[0] == 1.0
