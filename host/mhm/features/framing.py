"""Explicit windowing and spectra shared by the vibration and audio features."""

import numpy as np

EPS = 1e-12


def frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Overlapping windows along axis 0, shape (n_windows, size, *x.shape[1:])."""
    n = 0 if len(x) < size else 1 + (len(x) - size) // hop
    index = hop * np.arange(n)[:, None] + np.arange(size)[None, :]
    return x[index]


def hann(size: int) -> np.ndarray:
    """Periodic Hann window, the variant that sums to a constant at 50% overlap."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)


def power_spectrum(windows: np.ndarray) -> np.ndarray:
    """Hann-windowed |DFT|^2 along axis 1, keeping bins 0 to size/2."""
    size = windows.shape[1]
    window = hann(size).reshape((1, size) + (1,) * (windows.ndim - 2))
    # Plain DFT; the RTL swaps in a fixed-point radix-2 FFT that matches it up to rounding.
    return np.abs(np.fft.rfft(windows * window, axis=1)) ** 2
