from pathlib import Path

import numpy as np

from mhm.capture.frame import HEADER, StreamType, encode
from mhm.capture.loader import load_capture


def test_loader_reassembles_streams(tmp_path: Path):
    rng = np.random.default_rng(1)
    accel = rng.integers(-500, 500, size=(5, 32, 3), dtype=np.int16)
    audio = rng.integers(-30000, 30000, size=(3, 256), dtype=np.int16)

    chunks = [encode(StreamType.LOG, 0, 0, b"node up")]
    for i in range(5):
        if i == 3:
            continue  # dropped on the node, so accel seq 3 never arrives
        chunks.append(encode(StreamType.ACCEL, i, 2_000_000 + i * 10_000, accel[i].tobytes()))
    for i in range(3):
        chunks.append(encode(StreamType.AUDIO, i, 2_000_000 + i * 16_000, audio[i].tobytes()))
    corrupt = bytearray(encode(StreamType.AUDIO, 3, 2_048_000, audio[0].tobytes()))
    corrupt[HEADER.size] ^= 0xFF
    chunks.append(bytes(corrupt))

    path = tmp_path / "sim_normal_1500rpm_20261005_01.bin"
    path.write_bytes(b"garbage" + b"".join(chunks))
    cap = load_capture(path)

    kept = accel[[0, 1, 2, 4]].reshape(-1, 3)
    np.testing.assert_array_equal(cap.accel.samples, kept)
    assert cap.accel.fs == 3200.0
    assert cap.accel.t_us[0] == 2_000_000
    assert cap.accel.t_us[1] == 2_000_000 + 312  # 312.5 us rounds to even
    assert cap.accel.t_us[96] == 2_040_000  # first sample after the gap
    np.testing.assert_array_equal(cap.accel.breaks, [96])
    assert [len(s) for s in cap.accel.segments()] == [96, 32]

    np.testing.assert_array_equal(cap.audio.samples[:, 0], audio.reshape(-1))
    assert cap.audio.t_us[-1] == 2_032_000 + round(255 * 62.5)
    assert len(cap.audio.breaks) == 0

    assert cap.log == ["node up"]
    assert cap.crc_errors == 1
    assert cap.dropped == {StreamType.ACCEL: 1, StreamType.AUDIO: 0}


def test_loader_handles_missing_stream(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_bytes(encode(StreamType.LOG, 0, 0, b"adxl345 not found"))
    cap = load_capture(path)
    assert cap.accel.samples.shape == (0, 3)
    assert cap.audio.samples.shape == (0, 1)
    assert cap.accel.segments() == []
