import csv
import io
import itertools
import sys
from pathlib import Path

import pytest

from mhm.capture.loader import load_capture
from mhm.capture.record import CaptureName, record
from mhm.capture.simulate import FakeNode, SimConfig


def test_record_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frames = itertools.takewhile(lambda item: item[0] <= 1.0, FakeNode(SimConfig()).frames())
    data = b"".join(d for _, d in frames)
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


def test_capture_name_rejects_underscores():
    with pytest.raises(ValueError):
        CaptureName("fan_1", "normal", "1500rpm", 1)
