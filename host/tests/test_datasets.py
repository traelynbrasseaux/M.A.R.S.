import wave
from pathlib import Path

import numpy as np

from mhm.datasets import mafaulda, mimii


def write_wav(path: Path, data: np.ndarray, fs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f, wave.open(f, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data.astype("<i2").tobytes())


def test_mimii_listing_and_channel_select(tmp_path: Path):
    data = np.arange(800, dtype=np.int16).reshape(100, 8)
    for label, name in [
        ("normal", "00000001.wav"),
        ("normal", "00000000.wav"),
        ("abnormal", "00000000.wav"),
    ]:
        write_wav(tmp_path / "pump" / "id_00" / label / name, data, 16000)

    clips = mimii.list_clips(tmp_path)
    assert [(c.path.parent.name, c.path.name, c.abnormal) for c in clips] == [
        ("normal", "00000000.wav", False),
        ("normal", "00000001.wav", False),
        ("abnormal", "00000000.wav", True),
    ]
    x, fs = mimii.load_wav(clips[0].path, channel=3)
    assert fs == 16000
    np.testing.assert_array_equal(x, data[:, 3])


def test_mafaulda_decimation_keeps_low_and_removes_high(tmp_path: Path):
    t = np.arange(50000) / mafaulda.FS
    low = np.sin(2 * np.pi * 25 * t)
    high = np.sin(2 * np.pi * 5000 * t)  # far above the 1562 Hz decimated Nyquist
    table = np.zeros((len(t), 8))
    table[:, 1] = low + high
    table[:, 2] = low
    path = tmp_path / "normal" / "25.0.csv"
    path.parent.mkdir()
    np.savetxt(path, table, delimiter=",", fmt="%.6f")

    x, fs = mafaulda.load_vibration(path)
    assert fs == 3125.0
    assert x.shape[1] == 3
    np.testing.assert_allclose(x[:, 0], x[:, 1], atol=0.01)
    assert mafaulda.rotation_hz(path) == 25.0
    assert [c.abnormal for c in mafaulda.list_clips(tmp_path)] == [False]
