"""Per-window vibration features: FFT band energies, RMS, kurtosis, crest factor."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from mhm.features.framing import EPS, power_spectrum


@dataclass(frozen=True)
class VibrationConfig:
    fs: float = 3200.0
    size: int = 1024
    hop: int = 512
    band_edges_hz: tuple[float, ...] = (5, 15, 35, 60, 100, 200, 400, 800, 1600)

    @property
    def per_axis(self) -> int:
        return len(self.band_edges_hz) - 1 + 3


def band_energies(power: np.ndarray, freqs: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Sum power over bins with lo <= f < hi for each band; bands land on axis 1."""
    bands = [power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in pairwise(edges)]
    return np.stack(bands, axis=1)


def vibration_features(windows: np.ndarray, cfg: VibrationConfig) -> np.ndarray:
    """(n, size, axes) windows to per-axis [log10 band energies, RMS, kurtosis, crest], flat."""
    x = windows.astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)  # drops gravity and sensor offset
    power = power_spectrum(x)
    freqs = np.arange(power.shape[1]) * cfg.fs / cfg.size
    log_bands = np.log10(band_energies(power, freqs, cfg.band_edges_hz) + EPS)

    rms = np.sqrt(np.mean(x**2, axis=1))
    kurtosis = np.mean(x**4, axis=1) / np.maximum(rms**4, EPS)
    crest = np.max(np.abs(x), axis=1) / np.maximum(rms, EPS)

    per_axis = np.concatenate([log_bands, np.stack([rms, kurtosis, crest], axis=1)], axis=1)
    return per_axis.transpose(0, 2, 1).reshape(len(x), -1)
