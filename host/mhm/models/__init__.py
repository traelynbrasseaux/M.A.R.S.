"""Anomaly models fit on normal windows only; a higher score means more anomalous."""

from typing import Protocol, Self

import numpy as np


class AnomalyModel(Protocol):
    def fit(self, normal_windows: np.ndarray) -> Self: ...

    def score(self, windows: np.ndarray) -> np.ndarray: ...


def feature_std(x: np.ndarray) -> np.ndarray:
    """Per-feature std, with numerically constant features set to 1; lib/ml uses the same rule."""
    std = x.std(axis=0)
    constant = std**2 <= 1e-12 * np.maximum(1.0, x.mean(axis=0) ** 2)
    return np.where(constant, 1.0, std)
