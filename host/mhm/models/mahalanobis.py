"""Mahalanobis distance baseline."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from mhm.models import feature_std


@dataclass
class Mahalanobis:
    """Squared Mahalanobis distance on z-scored features, so the ridge is scale-free."""

    ridge: float = 1e-3
    mean: np.ndarray = field(init=False, repr=False)
    std: np.ndarray = field(init=False, repr=False)
    precision: np.ndarray = field(init=False, repr=False)

    def fit(self, normal_windows: np.ndarray) -> Self:
        x = np.asarray(normal_windows, np.float64)
        self.mean = x.mean(axis=0)
        self.std = feature_std(x)
        z = (x - self.mean) / self.std
        cov = z.T @ z / (len(z) - 1)
        # The ridge keeps the inverse stable when features are redundant, e.g. MFCCs of log mels.
        self.precision = np.linalg.inv(cov + self.ridge * np.eye(len(cov)))
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        z = (np.asarray(windows, np.float64) - self.mean) / self.std
        return np.einsum("ij,jk,ik->i", z, self.precision, z)
