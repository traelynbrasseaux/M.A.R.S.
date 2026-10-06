"""Anomaly models fit on normal windows only; a higher score means more anomalous."""

from typing import Protocol, Self

import numpy as np


class AnomalyModel(Protocol):
    def fit(self, normal_windows: np.ndarray) -> Self: ...

    def score(self, windows: np.ndarray) -> np.ndarray: ...
