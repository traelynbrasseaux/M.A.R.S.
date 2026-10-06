import itertools

import numpy as np

from mhm.capture.frame import FrameParser, SequenceTracker, StreamType
from mhm.capture.simulate import FakeNode, SimConfig


def simulated_bytes(cfg: SimConfig, seconds: float) -> bytes:
    frames = itertools.takewhile(lambda item: item[0] <= seconds, FakeNode(cfg).frames())
    return b"".join(data for _, data in frames)


def test_simulator_rates_and_sequence():
    parser, tracker = FrameParser(), SequenceTracker()
    frames = parser.feed(simulated_bytes(SimConfig(), 2.0))
    for f in frames:
        tracker.update(f)
    kinds = [f.stream for f in frames]
    assert kinds[0] == StreamType.LOG
    assert kinds.count(StreamType.ACCEL) == 200
    assert kinds.count(StreamType.AUDIO) == 125
    assert parser.crc_errors == 0
    assert sum(tracker.dropped.values()) == 0


def one_x_amplitude(cfg: SimConfig) -> float:
    parser = FrameParser()
    accel = [f for f in parser.feed(simulated_bytes(cfg, 4.0)) if f.stream == StreamType.ACCEL]
    x = np.frombuffer(b"".join(f.payload for f in accel), "<i2").reshape(-1, 3)[:, 0]
    spectrum = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), 1 / 3200)
    band = (freqs > 0.9 * cfg.rotation_hz) & (freqs < 1.1 * cfg.rotation_hz)
    return float(spectrum[band].max())


def test_unbalance_boosts_one_x():
    normal = one_x_amplitude(SimConfig(seed=3))
    unbalanced = one_x_amplitude(SimConfig(seed=3, unbalance=True))
    assert unbalanced > 3 * normal


def test_fault_injection_shows_up_in_parser():
    parser, tracker = FrameParser(), SequenceTracker()
    cfg = SimConfig(drop_rate=0.05, corrupt_rate=0.05, seed=7)
    for f in parser.feed(simulated_bytes(cfg, 5.0)):
        tracker.update(f)
    assert parser.crc_errors > 0
    assert sum(tracker.dropped.values()) > parser.crc_errors
