import numpy as np
import pytest

from mhm.capture.frame import (
    HEADER,
    Frame,
    FrameParser,
    SequenceTracker,
    StreamType,
    crc16,
    encode,
)


def accel_payload(n: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, size=(n, 3), dtype="<i2").tobytes()


def test_crc16_check_value():
    assert crc16(b"123456789") == 0x29B1


def test_round_trip_all_streams():
    sent = [
        Frame(StreamType.ACCEL, 7, 1_000_000, 32, accel_payload(32)),
        Frame(StreamType.AUDIO, 3, 1_002_000, 4, np.array([1, -1, 32767, -32768], "<i2").tobytes()),
        Frame(StreamType.LOG, 0, 1_003_000, 5, b"hello"),
    ]
    stream = b"".join(encode(f.stream, f.seq, f.timestamp_us, f.payload) for f in sent)
    parser = FrameParser()
    assert parser.feed(stream) == sent
    assert parser.crc_errors == 0
    assert parser.skipped_bytes == 0


def test_split_chunks_and_garbage():
    frames = [encode(StreamType.ACCEL, i, 1000 * i, accel_payload(32, i)) for i in range(5)]
    stream = b"boot noise \xa5\xa5\x5a" + frames[0] + b"\x00\xa5" + b"".join(frames[1:])
    parser = FrameParser()
    out = []
    for i in range(0, len(stream), 7):
        out += parser.feed(stream[i : i + 7])
    assert [f.seq for f in out] == [0, 1, 2, 3, 4]
    assert parser.crc_errors == 0


def test_crc_failure_drops_only_that_frame():
    good = [encode(StreamType.AUDIO, i, 0, bytes(64)) for i in range(3)]
    bad = bytearray(good[1])
    bad[HEADER.size + 10] ^= 0x01
    parser = FrameParser()
    out = parser.feed(good[0] + bytes(bad) + good[2])
    assert [f.seq for f in out] == [0, 2]
    assert parser.crc_errors == 1


def test_encode_rejects_partial_sample():
    with pytest.raises(ValueError):
        encode(StreamType.ACCEL, 0, 0, bytes(7))


def frame(stream: StreamType, seq: int) -> Frame:
    return Frame(stream, seq, 0, 0, b"")


def test_sequence_gaps_per_stream():
    tracker = SequenceTracker()
    gaps = [
        tracker.update(frame(StreamType.ACCEL, 0)),
        tracker.update(frame(StreamType.AUDIO, 10)),
        tracker.update(frame(StreamType.ACCEL, 1)),
        tracker.update(frame(StreamType.ACCEL, 4)),
        tracker.update(frame(StreamType.AUDIO, 11)),
        tracker.update(frame(StreamType.AUDIO, 13)),
    ]
    assert gaps == [0, 0, 0, 2, 0, 1]
    assert tracker.dropped == {StreamType.ACCEL: 2, StreamType.AUDIO: 1}


def test_sequence_wraparound_and_restart():
    tracker = SequenceTracker()
    tracker.update(frame(StreamType.ACCEL, 0xFFFFFFFF))
    assert tracker.update(frame(StreamType.ACCEL, 0)) == 0
    assert tracker.update(frame(StreamType.ACCEL, 500)) == 499
    assert tracker.update(frame(StreamType.ACCEL, 0)) == 0
    assert tracker.restarts == 1
    assert tracker.dropped[StreamType.ACCEL] == 499
