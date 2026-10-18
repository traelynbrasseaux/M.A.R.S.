"""Run the node's C++ ML (firmware/node/lib/ml) on the PC through its native PlatformIO build."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from mhm import REPO_ROOT

FIRMWARE_DIR = REPO_ROOT / "firmware" / "node"
NATIVE_BUILD = FIRMWARE_DIR / ".pio" / "build" / "native"
CLI = NATIVE_BUILD / ("program.exe" if sys.platform == "win32" else "program")
FEATURE_WIDTH = {"accel": 33, "audio": 45}

Kind = Literal["accel", "audio"]


def build() -> Path:
    """Build or refresh the native CLI; PlatformIO skips the work when nothing changed."""
    pio = shutil.which("pio")
    if pio is None:
        raise FileNotFoundError("pio is not on PATH, so firmware/node cannot be built for the PC")
    subprocess.run([pio, "run", "-s", "-e", "native", "-d", str(FIRMWARE_DIR)], check=True)
    return CLI


def run(*args: object) -> None:
    if not CLI.exists():
        raise FileNotFoundError(f"{CLI} is missing; run `pio run -e native -d firmware/node`")
    subprocess.run([str(CLI), *map(str, args)], check=True)


def features(kind: Kind, signal: np.ndarray, fs: float) -> np.ndarray:
    """Per-window features from the C++ extractor, (n_windows, 33) for accel or (n_windows, 45)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.f32", Path(tmp) / "out.f32"
        np.asarray(signal, "<f4").tofile(src)
        run("features", kind, fs, src, dst)
        return np.fromfile(dst, "<f4").reshape(-1, FEATURE_WIDTH[kind]).astype(np.float64)


def scores(stats: np.ndarray, bundle: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Learn in the node's two phases, then score test windows: mahalanobis, hdc binary, bipolar."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"{name}.f32" for name in ("stats", "bundle", "test", "out")]
        for path, x in zip(paths, (stats, bundle, test), strict=False):
            np.asarray(x, "<f4").tofile(path)
        run("score", stats.shape[1], *paths)
        return np.fromfile(paths[3], "<f4").reshape(-1, 3).astype(np.float64)
