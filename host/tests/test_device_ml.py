"""The node's C++ ML, built for the PC, must agree with the Python golden model."""

import itertools
import subprocess

import numpy as np
import pytest

from mhm import device_ml
from mhm.capture.frame import FrameParser, StreamType
from mhm.capture.simulate import FakeNode, SimConfig
from mhm.features.audio import AudioConfig, audio_features
from mhm.features.framing import frames
from mhm.features.vibration import VibrationConfig, vibration_features


@pytest.fixture(scope="module", autouse=True)
def native_cli():
    try:
        device_ml.build()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"native build of firmware/node unavailable: {e}")


@pytest.fixture(scope="module")
def simulated() -> tuple[np.ndarray, np.ndarray]:
    stream = itertools.takewhile(lambda t: t[0] <= 6, FakeNode(SimConfig(unbalance=True)).frames())
    parsed = FrameParser().feed(b"".join(d for _, d in stream))
    accel = b"".join(f.payload for f in parsed if f.stream == StreamType.ACCEL)
    audio = b"".join(f.payload for f in parsed if f.stream == StreamType.AUDIO)
    return np.frombuffer(accel, "<i2").reshape(-1, 3), np.frombuffer(audio, "<i2")


def test_vibration_features_match(simulated):
    cfg = VibrationConfig()
    expected = vibration_features(frames(simulated[0], cfg.size, cfg.hop), cfg)
    np.testing.assert_allclose(
        device_ml.features("accel", simulated[0], cfg.fs), expected, atol=1e-4
    )


def test_vibration_features_match_at_mafaulda_rate():
    cfg = VibrationConfig(fs=3125.0)
    x = np.random.default_rng(0).normal(0, 0.3, size=(8 * cfg.size, 3))
    expected = vibration_features(frames(x, cfg.size, cfg.hop), cfg)
    np.testing.assert_allclose(device_ml.features("accel", x, cfg.fs), expected, atol=1e-4)


def test_audio_features_match(simulated):
    cfg = AudioConfig()
    expected = audio_features(frames(simulated[1], cfg.size, cfg.hop), cfg)
    np.testing.assert_allclose(
        device_ml.features("audio", simulated[1], cfg.fs), expected, atol=1e-4
    )
