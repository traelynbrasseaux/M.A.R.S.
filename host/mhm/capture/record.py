"""Record a node's UART stream to a .bin log and index it in captures.csv."""

import csv
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import serial

from mhm.capture.frame import FrameParser, SequenceTracker, StreamType

BAUD = 921600
TOKEN = re.compile(r"[a-z0-9-]+")
INDEX_COLUMNS = [
    "file",
    "machine",
    "condition",
    "speed",
    "date",
    "run",
    "start_utc",
    "duration_s",
    "bytes",
    "accel_frames",
    "audio_frames",
    "crc_errors",
    "dropped_accel",
    "dropped_audio",
    "port",
    "notes",
]


@dataclass(frozen=True)
class CaptureName:
    machine: str
    condition: str
    speed: str
    run: int
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))

    def __post_init__(self):
        for key in ("machine", "condition", "speed"):
            if not TOKEN.fullmatch(value := getattr(self, key)):
                raise ValueError(f"{key} may only use a-z, 0-9 and '-', got {value!r}")
        if not 1 <= self.run <= 99:
            raise ValueError(f"run must be 1 to 99, got {self.run}")

    @property
    def filename(self) -> str:
        return f"{self.machine}_{self.condition}_{self.speed}_{self.date}_{self.run:02d}.bin"


@dataclass
class LiveStats:
    parser: FrameParser = field(default_factory=FrameParser)
    tracker: SequenceTracker = field(default_factory=SequenceTracker)
    frames: Counter[StreamType] = field(default_factory=Counter)
    bytes: int = 0

    def summary(self, elapsed: float, rates: dict[StreamType, float]) -> str:
        dropped = sum(self.tracker.dropped.values())
        return (
            f"{elapsed:7.1f} s | accel {rates.get(StreamType.ACCEL, 0):6.1f} fps"
            f" | audio {rates.get(StreamType.AUDIO, 0):5.1f} fps"
            f" | crc errors {self.parser.crc_errors} | dropped {dropped}"
            f" | {self.bytes / 1e6:.2f} MB"
        )


def record(
    port: str,
    name: CaptureName,
    out_dir: Path,
    duration: float | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Capture until duration, Ctrl+C, or end of stdin; return the captures.csv row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name.filename
    if path.exists():
        raise FileExistsError(f"{path} already exists, use the next --run number")

    ser = None
    if port != "-":
        ser = serial.serial_for_url(port, baudrate=BAUD, timeout=0.1, do_not_open=True)
        # Low DTR and RTS keep the board's auto-reset circuit from rebooting the ESP32 on open.
        ser.dtr = ser.rts = False
        ser.open()
    stats = LiveStats()
    start_utc = datetime.now(UTC)
    start = last_report = time.monotonic()
    last_frames: Counter[StreamType] = Counter()
    try:
        with path.open("wb") as f:
            while duration is None or time.monotonic() - start < duration:
                if ser is None:
                    data = sys.stdin.buffer.read1(65536)
                else:
                    data = ser.read(ser.in_waiting or 1)
                if not data:
                    if ser is None:
                        break  # stdin closed
                    continue  # serial read timed out
                f.write(data)
                stats.bytes += len(data)
                for frame in stats.parser.feed(data):
                    stats.tracker.update(frame)
                    stats.frames[frame.stream] += 1
                    if frame.stream == StreamType.LOG:
                        text = frame.payload.decode("utf-8", errors="replace")
                        print(f"\r[node] {text}".ljust(100))
                now = time.monotonic()
                if (dt := now - last_report) >= 1.0:
                    rates = {k: (stats.frames[k] - last_frames[k]) / dt for k in StreamType}
                    print("\r" + stats.summary(now - start, rates), end="", flush=True)
                    last_frames, last_report = stats.frames.copy(), now
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            ser.close()

    elapsed = time.monotonic() - start
    average = {k: stats.frames[k] / max(elapsed, 1e-9) for k in StreamType}
    print("\r" + stats.summary(elapsed, average))
    print(f"wrote {path}")

    row = {
        "file": path.name,
        "machine": name.machine,
        "condition": name.condition,
        "speed": name.speed,
        "date": name.date,
        "run": name.run,
        "start_utc": start_utc.isoformat(timespec="seconds"),
        "duration_s": round(elapsed, 2),
        "bytes": stats.bytes,
        "accel_frames": stats.frames[StreamType.ACCEL],
        "audio_frames": stats.frames[StreamType.AUDIO],
        "crc_errors": stats.parser.crc_errors,
        "dropped_accel": stats.tracker.dropped.get(StreamType.ACCEL, 0),
        "dropped_audio": stats.tracker.dropped.get(StreamType.AUDIO, 0),
        "port": port,
        "notes": notes,
    }
    index = out_dir / "captures.csv"
    new_index = not index.exists()
    with index.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new_index:
            writer.writeheader()
        writer.writerow(row)
    return row
