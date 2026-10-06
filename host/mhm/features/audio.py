"""Per-window audio features: log mel band energies and MFCCs."""

from dataclasses import dataclass

import numpy as np

from mhm.features.framing import EPS, power_spectrum


@dataclass(frozen=True)
class AudioConfig:
    fs: float = 16000.0
    size: int = 1024
    hop: int = 512
    n_mels: int = 32
    n_mfcc: int = 13
    fmin: float = 50.0
    fmax: float = 8000.0


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(cfg: AudioConfig) -> np.ndarray:
    """Triangular filters with unit peaks on the HTK mel scale, (n_mels, size // 2 + 1)."""
    freqs = np.arange(cfg.size // 2 + 1) * cfg.fs / cfg.size
    edges = mel_to_hz(np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2))
    lo, center, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs - lo) / (center - lo)
    falling = (hi - freqs) / (hi - center)
    return np.maximum(0.0, np.minimum(rising, falling))


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II basis, one row per output coefficient."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    basis[0] /= np.sqrt(2.0)
    return basis


def audio_features(windows: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """(n, size) int16 windows to (n, n_mels + n_mfcc) features: log10 mel energies, then MFCCs."""
    x = windows.astype(np.float64) / 32768.0
    x -= x.mean(axis=1, keepdims=True)
    log_mel = np.log10(power_spectrum(x) @ mel_filterbank(cfg).T + EPS)
    mfcc = log_mel @ dct_matrix(cfg.n_mfcc, cfg.n_mels).T
    return np.concatenate([log_mel, mfcc], axis=1)
