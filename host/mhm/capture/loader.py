"""Read a raw .bin capture back into per-stream numpy arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mhm.capture.frame import (
    ACCEL_FS,
    AUDIO_FS,
    SCORE,
    Frame,
    FrameParser,
    SequenceTracker,
    StreamType,
)

CHANNELS = {StreamType.ACCEL: 3, StreamType.AUDIO: 1}
SAMPLE_RATE = {StreamType.ACCEL: ACCEL_FS, StreamType.AUDIO: AUDIO_FS}


@dataclass
class Stream:
    fs: float
    samples: np.ndarray  # (n, channels) int16; accel channels are x, y, z
    t_us: np.ndarray  # (n,) int64 estimated acquisition time of each sample
    breaks: np.ndarray  # sample indices where missing frames start a new segment

    def segments(self) -> list[np.ndarray]:
        """Contiguous runs of samples, split wherever frames are missing."""
        return [s for s in np.split(self.samples, self.breaks) if len(s)]


@dataclass
class Scores:
    """The node's per-block detector results, one row per score frame."""

    t_us: np.ndarray  # node time each result was sent
    stream: np.ndarray  # StreamType.ACCEL or StreamType.AUDIO
    phase: np.ndarray  # index into frame.PHASES
    alerts: np.ndarray  # (n, 3) bool, columns as frame.MODELS
    scores: np.ndarray  # (n, 3)
    thresholds: np.ndarray  # (n, 3), 0 until calibration ends


@dataclass
class Capture:
    accel: Stream
    audio: Stream
    scores: Scores
    log: list[str]
    crc_errors: int
    dropped: dict[StreamType, int]


def scores_from_frames(frames: list[Frame]) -> Scores:
    rows = np.array([SCORE.unpack(f.payload) for f in frames], np.float64).reshape(-1, 9)
    bits = rows[:, 2].astype(int)
    return Scores(
        t_us=np.array([f.timestamp_us for f in frames], np.int64),
        stream=rows[:, 0].astype(int),
        phase=rows[:, 1].astype(int),
        alerts=np.stack([(bits >> m) & 1 for m in range(3)], axis=1).astype(bool),
        scores=rows[:, 3:6],
        thresholds=rows[:, 6:9],
    )


def stream_from_frames(frames: list[Frame], kind: StreamType) -> Stream:
    fs, channels = SAMPLE_RATE[kind], CHANNELS[kind]
    if not frames:
        empty = np.empty(0, np.int64)
        return Stream(fs, np.empty((0, channels), np.int16), empty, empty)
    raw = b"".join(f.payload for f in frames)
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.int16)

    counts = np.array([f.sample_count for f in frames])
    frame_start = np.cumsum(counts) - counts
    index_in_frame = np.arange(len(samples)) - np.repeat(frame_start, counts)
    t0 = np.repeat(np.array([f.timestamp_us for f in frames], np.int64), counts)
    t_us = t0 + np.round(index_in_frame * 1e6 / fs).astype(np.int64)

    seqs = np.array([f.seq for f in frames], np.int64)
    gap = (np.diff(seqs) & 0xFFFFFFFF) != 1
    return Stream(fs, samples, t_us, frame_start[1:][gap])


def load_capture(path: Path) -> Capture:
    parser = FrameParser()
    tracker = SequenceTracker()
    frames = parser.feed(path.read_bytes())
    for f in frames:
        tracker.update(f)
    by_kind = {kind: [f for f in frames if f.stream == kind] for kind in StreamType}
    return Capture(
        accel=stream_from_frames(by_kind[StreamType.ACCEL], StreamType.ACCEL),
        audio=stream_from_frames(by_kind[StreamType.AUDIO], StreamType.AUDIO),
        scores=scores_from_frames(by_kind[StreamType.SCORE]),
        log=[f.payload.decode("utf-8", errors="replace") for f in by_kind[StreamType.LOG]],
        crc_errors=parser.crc_errors,
        dropped=tracker.dropped,
    )
