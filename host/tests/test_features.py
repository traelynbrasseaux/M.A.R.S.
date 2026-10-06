import numpy as np

from mhm.features.audio import AudioConfig, audio_features, dct_matrix, mel_filterbank
from mhm.features.framing import frames, hann
from mhm.features.vibration import VibrationConfig, vibration_features


def test_frames_and_hann():
    np.testing.assert_array_equal(frames(np.arange(9), 4, 3), [[0, 1, 2, 3], [3, 4, 5, 6]])
    assert frames(np.zeros((3, 2)), 4, 2).shape == (0, 4, 2)
    np.testing.assert_allclose(hann(4), [0.0, 0.5, 1.0, 0.5], atol=1e-12)


def test_vibration_features_on_known_signals():
    cfg = VibrationConfig()
    t = np.arange(cfg.size * 4) / cfg.fs
    rng = np.random.default_rng(0)
    x = 1000 * np.sin(2 * np.pi * 25 * t)  # sine on x
    y = rng.normal(0, 200, size=t.shape)  # noise on y
    z = np.full(t.shape, 256.0)  # gravity only on z
    windows = frames(np.stack([x, y, z], axis=1), cfg.size, cfg.hop)
    feats = vibration_features(windows, cfg)

    n_bands = len(cfg.band_edges_hz) - 1
    assert feats.shape == (len(windows), 3 * cfg.per_axis)
    fx, fy = feats[:, : cfg.per_axis], feats[:, cfg.per_axis : 2 * cfg.per_axis]
    assert np.all(np.argmax(fx[:, :n_bands], axis=1) == 1)  # the 15 to 35 Hz band
    rms, kurtosis, crest = fx[:, n_bands], fx[:, n_bands + 1], fx[:, n_bands + 2]
    np.testing.assert_allclose(rms, 1000 / np.sqrt(2), rtol=0.01)
    np.testing.assert_allclose(kurtosis, 1.5, rtol=0.02)
    np.testing.assert_allclose(crest, np.sqrt(2), rtol=0.01)
    assert np.all(np.abs(fy[:, n_bands + 1] - 3.0) < 0.6)  # Gaussian kurtosis
    assert np.all(feats[:, 2 * cfg.per_axis + n_bands] == 0)  # z RMS after removing gravity


def test_mel_filterbank_and_dct():
    cfg = AudioConfig()
    bank = mel_filterbank(cfg)
    assert bank.shape == (cfg.n_mels, cfg.size // 2 + 1)
    assert np.all(bank.sum(axis=1) > 0)  # no empty filters
    np.testing.assert_allclose(dct_matrix(8, 8) @ dct_matrix(8, 8).T, np.eye(8), atol=1e-12)


def test_audio_tone_lands_in_matching_mel_band():
    cfg = AudioConfig()
    t = np.arange(cfg.size * 3) / cfg.fs
    tone = np.round(8000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    feats = audio_features(frames(tone, cfg.size, cfg.hop), cfg)
    assert feats.shape == (5, cfg.n_mels + cfg.n_mfcc)
    peak = np.argmax(feats[:, : cfg.n_mels], axis=1)
    centers = np.argmax(mel_filterbank(cfg), axis=1) * cfg.fs / cfg.size
    assert np.all(np.abs(centers[peak] - 1000) < 100)
