import csv
import io
import itertools
import sys
from pathlib import Path

import pytest

from mhm.capture import record as record_module
from mhm.capture.frame import PHASES, SCORE, StreamType
from mhm.capture.loader import load_capture
from mhm.capture.record import CaptureName, LiveStats, record
from mhm.capture.simulate import FakeNode, SimConfig


def simulated(seconds: float) -> bytes:
    frames = itertools.takewhile(lambda item: item[0] <= seconds, FakeNode(SimConfig()).frames())
    return b"".join(d for _, d in frames)


def test_record_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = simulated(1.0)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BufferedReader(io.BytesIO(data))))
    name = CaptureName("sim", "normal", "1500rpm", 1, date="20261005")
    row = record("-", name, tmp_path)

    assert row["file"] == "sim_normal_1500rpm_20261005_01.bin"
    assert row["accel_frames"] == 100
    assert row["audio_frames"] == 62
    assert (tmp_path / row["file"]).read_bytes() == data
    with (tmp_path / "captures.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["file"] == row["file"]
    assert load_capture(tmp_path / row["file"]).accel.samples.shape == (3200, 3)

    with pytest.raises(FileExistsError):
        record("-", name, tmp_path)


class FakePort:
    """Just enough of a pyserial port: hands out canned bytes and keeps what was written."""

    def __init__(self, data: bytes):
        self.data, self.written = data, b""
        self.dtr = self.rts = True

    def open(self):
        pass

    def close(self):
        pass

    @property
    def in_waiting(self) -> int:
        return len(self.data)

    def read(self, n: int) -> bytes:
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk

    def write(self, b: bytes) -> int:
        self.written += b
        return len(b)


def test_record_sends_commands_and_notes_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    port = FakePort(simulated(0.5))
    monkeypatch.setattr(record_module.serial, "serial_for_url", lambda *a, **k: port)
    name = CaptureName("fan1", "normal", "1500rpm", 1, date="20261018")
    row = record("COM9", name, tmp_path, duration=0.3, notes="bench", commands=("learn 60 30",))

    assert port.written == b"learn 60 30\n"
    assert port.dtr is False and port.rts is False
    assert row["notes"] == "bench; sent learn 60 30"
    assert row["accel_frames"] == 50


def test_send_needs_a_serial_port(tmp_path: Path):
    with pytest.raises(ValueError):
        record("-", CaptureName("sim", "normal", "0rpm", 1), tmp_path, commands=("status",))


def test_ml_state_lines_only_on_change():
    stats = LiveStats()
    calibrate = SCORE.pack(StreamType.ACCEL, PHASES.index("calibrate"), 0, *[0.0] * 6)
    alert = SCORE.pack(StreamType.ACCEL, PHASES.index("detect"), 0b011, *[1.0] * 6)
    assert stats.ml_change(calibrate) == "[ml] accel calibrate"
    assert stats.ml_change(calibrate) is None
    assert stats.ml_change(alert) == "[ml] accel detect ALERT mahalanobis, hdc-binary"


def test_capture_name_rejects_underscores():
    with pytest.raises(ValueError):
        CaptureName("fan_1", "normal", "1500rpm", 1)
