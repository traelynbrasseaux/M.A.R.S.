"""Run the node's C++ ML (firmware/node/lib/ml) on the PC through its native PlatformIO build."""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mhm import REPO_ROOT
from mhm.capture.frame import MODELS, PHASES

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


@dataclass
class Replay:
    t: np.ndarray  # seconds of signal at the end of each ~1 s block
    phase: np.ndarray  # index into PHASES
    alerts: np.ndarray  # (blocks, 3) bool, columns as MODELS
    scores: np.ndarray  # (blocks, 3) mean window score per block
    thresholds: np.ndarray  # (blocks, 3), 0 until calibration ends


def replay(
    kind: Kind, samples: np.ndarray, learn_s: float, calibrate_s: float, margin: float = 1.2
) -> Replay:
    """Run raw int16 samples through the node's own detector, learning from the first sample."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.i16", Path(tmp) / "out.f32"
        np.asarray(samples, "<i2").tofile(src)
        run("replay", kind, src, learn_s, calibrate_s, margin, dst)
        rows = np.fromfile(dst, "<f4").reshape(-1, 9).astype(np.float64)
    alerts = rows[:, 2].astype(int)
    return Replay(
        t=rows[:, 0],
        phase=rows[:, 1].astype(int),
        alerts=np.stack([(alerts >> m) & 1 for m in range(3)], axis=1).astype(bool),
        scores=rows[:, 3:6],
        thresholds=rows[:, 6:9],
    )


def summarize(r: Replay, offset_s: float = 0.0) -> str:
    """Phase lengths, then each model's threshold and alert rate while detecting."""
    present = [(i, name) for i, name in enumerate(PHASES) if np.any(r.phase == i)]
    lines = ["  blocks: " + ", ".join(f"{name} {np.sum(r.phase == i)}" for i, name in present)]
    detect = r.phase == PHASES.index("detect")
    if not detect.any():
        lines.append("  never reached detect: recording shorter than learning plus calibration")
        return "\n".join(lines)
    for m, model in enumerate(MODELS):
        alerts = r.alerts[detect, m]
        first = f", first at {r.t[detect][alerts][0] + offset_s:.1f} s" if alerts.any() else ""
        lines.append(
            f"  {model:<12} threshold {r.thresholds[detect][0, m]:.4g},"
            f" alerts in {alerts.mean():.0%} of {detect.sum()} detect blocks{first}"
        )
    return "\n".join(lines)
