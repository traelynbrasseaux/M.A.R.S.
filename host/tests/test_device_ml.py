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
from mhm.models.hdc import HdcScorer
from mhm.models.mahalanobis import Mahalanobis


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


@pytest.mark.parametrize("kind", ["accel", "audio"])
def test_models_match(simulated, kind):
    signal = simulated[0] if kind == "accel" else simulated[1]
    fs = 3200.0 if kind == "accel" else 16000.0
    # Round to float32 first so both sides see exactly the same feature values.
    x = device_ml.features(kind, signal, fs).astype(np.float32).astype(np.float64)
    third = len(x) // 3
    stats, bundle, test = x[:third], x[third : 2 * third], x[2 * third :]
    got = device_ml.scores(stats, bundle, test)

    mahalanobis = Mahalanobis().fit(np.vstack([stats, bundle])).score(test)
    binary = HdcScorer(mode="binary").fit_quantizer(stats).bundle(bundle).score(test)
    bipolar = HdcScorer(mode="bipolar").fit_quantizer(stats).bundle(bundle).score(test)
    np.testing.assert_allclose(got[:, 0], mahalanobis, rtol=1e-3)
    np.testing.assert_array_equal(got[:, 1], binary.astype(np.float32))
    np.testing.assert_allclose(got[:, 2], bipolar, atol=1e-6)
