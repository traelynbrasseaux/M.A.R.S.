"""MAFAULDA: 5 s, 50 kHz CSV records from a SpectraQuest machinery fault simulator (UFRJ)."""

from pathlib import Path

import numpy as np

from mhm.datasets import Clip

URL = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/{name}.zip"
FS = 50000.0
# Columns: tachometer, underhang accel (axial, radial, tangential), overhang accel (same), mic.
BEARING_COLUMNS = {"underhang": (1, 2, 3), "overhang": (4, 5, 6)}


def list_clips(root: Path, fault: str = "imbalance") -> list[Clip]:
    """Records under root/normal/ and root/{fault}/**/, normal first."""
    normal = sorted((root / "normal").glob("*.csv"))
    abnormal = sorted((root / fault).rglob("*.csv"))
    return [Clip(p, False) for p in normal] + [Clip(p, True) for p in abnormal]


def rotation_hz(path: Path) -> float:
    """MAFAULDA names each record after its measured shaft frequency, e.g. 12.288.csv."""
    return float(path.stem)


def lowpass_taps(factor: int, taps_per_phase: int = 8) -> np.ndarray:
    """Hamming-windowed sinc with its cutoff at 90% of the decimated Nyquist frequency."""
    n = taps_per_phase * factor + 1
    cutoff = 0.9 * 0.5 / factor
    k = np.arange(n) - (n - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.hamming(n)
    return h / h.sum()


def load_vibration(
    path: Path, bearing: str = "underhang", factor: int = 16
) -> tuple[np.ndarray, float]:
    """Three-axis bearing accelerometer, decimated; factor 16 gives 3125 Hz, near the node."""
    data = np.loadtxt(path, delimiter=",", usecols=BEARING_COLUMNS[bearing])
    h = lowpass_taps(factor)
    out = np.stack([np.convolve(data[:, i], h, mode="valid")[::factor] for i in range(3)], axis=1)
    return out, FS / factor
