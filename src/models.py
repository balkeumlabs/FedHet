"""Lightweight logistic-regression primitives for federated training (NumPy only).

A linear model is deliberate: it federates cleanly via FedAvg, trains in
milliseconds on a consumer CPU, and lets us mask individual features exactly to
the device a home owns -- which is the whole point of the study. No GPU, no LLM.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearModel:
    w: np.ndarray  # (D,)
    b: float

    def copy(self) -> "LinearModel":
        return LinearModel(self.w.copy(), float(self.b))

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(X))


def init_model(dim: int) -> LinearModel:
    return LinearModel(np.zeros(dim, dtype=np.float64), 0.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def local_train(model: LinearModel, X: np.ndarray, y: np.ndarray,
                mask: np.ndarray, epochs: int, lr: float, l2: float,
                ) -> LinearModel:
    """Full-batch gradient descent on one client.

    `mask` (D,) is 1 for features this client's devices observe, 0 otherwise.
    Unobserved feature columns of X are assumed already zeroed; we additionally
    zero their gradient so a home never updates a weight it cannot inform.
    """
    m = model.copy()
    n = len(y)
    for _ in range(epochs):
        p = _sigmoid(X @ m.w + m.b)
        g = p - y
        grad_w = (X.T @ g) / n + l2 * m.w
        grad_b = float(g.mean())
        grad_w *= mask                      # do not move unobserved weights
        m.w -= lr * grad_w
        m.b -= lr * grad_b
    return m
