"""UART frame format shared with the node firmware, specified in docs/interface.md."""

import binascii
import struct
from dataclasses import dataclass, field
from enum import IntEnum

MAGIC = b"\xa5\x5a"
VERSION = 1
HEADER = struct.Struct("<2sBBIQH")  # magic, version, stream, seq, timestamp_us, sample_count
TRAILER = struct.Struct("<H")
MAX_SAMPLES = 1024

ACCEL_FS = 3200.0
AUDIO_FS = 16000.0
ACCEL_G_PER_LSB = 0.0039


class StreamType(IntEnum):
    ACCEL = 1
    AUDIO = 2
    LOG = 3


BYTES_PER_SAMPLE = {StreamType.ACCEL: 6, StreamType.AUDIO: 2, StreamType.LOG: 1}


def crc16(data: bytes | bytearray) -> int:
    """CRC-16/CCITT-FALSE, which is exactly what binascii.crc_hqx computes with init 0xFFFF."""
    return binascii.crc_hqx(data, 0xFFFF)


@dataclass(frozen=True, slots=True)
class Frame:
    stream: StreamType
    seq: int
    timestamp_us: int
    sample_count: int
    payload: bytes


def encode(stream: StreamType, seq: int, timestamp_us: int, payload: bytes) -> bytes:
    count, rem = divmod(len(payload), BYTES_PER_SAMPLE[stream])
    if rem or count > MAX_SAMPLES:
        raise ValueError(f"bad {stream.name} payload length {len(payload)}")
    body = HEADER.pack(MAGIC, VERSION, stream, seq & 0xFFFFFFFF, timestamp_us, count) + payload
    return body + TRAILER.pack(crc16(body))


@dataclass
class FrameParser:
    """Incremental parser that resyncs on the magic and drops frames with a bad CRC."""

    crc_errors: int = 0
    skipped_bytes: int = 0
    _buf: bytearray = field(default_factory=bytearray, repr=False)

    def feed(self, data: bytes) -> list[Frame]:
        buf = self._buf
        buf += data
        frames = []
        pos = 0
        while True:
            start = buf.find(MAGIC, pos)
            if start < 0:
                # A trailing first magic byte may be completed by the next chunk.
                keep = 1 if buf.endswith(MAGIC[:1]) else 0
                self.skipped_bytes += len(buf) - pos - keep
                pos = len(buf) - keep
                break
            self.skipped_bytes += start - pos
            if len(buf) - start < HEADER.size:
                pos = start
                break
            _, version, stream, seq, timestamp_us, count = HEADER.unpack_from(buf, start)
            if version != VERSION or stream not in BYTES_PER_SAMPLE or count > MAX_SAMPLES:
                self.skipped_bytes += 1
                pos = start + 1
                continue
            end = start + HEADER.size + count * BYTES_PER_SAMPLE[stream] + TRAILER.size
            if len(buf) < end:
                pos = start
                break
            (crc,) = TRAILER.unpack_from(buf, end - TRAILER.size)
            if crc != crc16(buf[start : end - TRAILER.size]):
                self.crc_errors += 1
                self.skipped_bytes += 1
                pos = start + 1
                continue
            payload = bytes(buf[start + HEADER.size : end - TRAILER.size])
            frames.append(Frame(StreamType(stream), seq, timestamp_us, count, payload))
            pos = end
        del buf[:pos]
        return frames


@dataclass
class SequenceTracker:
    """Counts frames missing from gaps in each stream's sequence numbers."""

    dropped: dict[StreamType, int] = field(default_factory=dict)
    restarts: int = 0
    _last: dict[StreamType, int] = field(default_factory=dict, repr=False)

    def update(self, frame: Frame) -> int:
        """Return how many frames of this stream went missing just before `frame`."""
        last = self._last.get(frame.stream)
        self._last[frame.stream] = frame.seq
        if last is None:
            return 0
        gap = (frame.seq - last - 1) & 0xFFFFFFFF
        if gap >= 1 << 31:
            # Sequence went backwards, so the node rebooted.
            self.restarts += 1
            return 0
        self.dropped[frame.stream] = self.dropped.get(frame.stream, 0) + gap
        return gap
