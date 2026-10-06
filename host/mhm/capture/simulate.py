"""Fake node that streams valid frames carrying synthetic vibration and sound."""

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import serial

from mhm.capture.frame import ACCEL_FS, ACCEL_G_PER_LSB, AUDIO_FS, StreamType, encode

ACCEL_PER_FRAME = 32
AUDIO_PER_FRAME = 256
BOOT_US = 1_500_000
BAUD = 921600

HARMONIC_G = np.array([0.05, 0.03, 0.015])  # 1x, 2x, 3x shaft order amplitudes
AXIS_GAIN = np.array([1.0, 0.8, 0.4])  # x and y radial, z axial
ACCEL_NOISE_G = 0.02
BLADE_TONES = np.array([0.05, 0.03, 0.02, 0.01])  # blade pass harmonics, fraction of full scale
AUDIO_NOISE = 0.01
SPEED_WANDER = 0.01  # +/-1% speed drift
WANDER_PERIOD_S = 37.0


@dataclass(frozen=True)
class SimConfig:
    rotation_hz: float = 25.0
    unbalance: bool = False
    unbalance_gain: float = 4.0
    blades: int = 5
    seed: int = 0
    drop_rate: float = 0.0
    corrupt_rate: float = 0.0


class FakeNode:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.accel_phase = self.rng.uniform(0, 2 * np.pi, size=(len(HARMONIC_G), 3))
        self.audio_phase = self.rng.uniform(0, 2 * np.pi, size=len(BLADE_TONES))
        self.seq = dict.fromkeys(StreamType, 0)

    def shaft_phase(self, t: np.ndarray) -> np.ndarray:
        """Shaft angle in radians, with the speed wandering slowly around rotation_hz."""
        w = 2 * np.pi / WANDER_PERIOD_S
        return 2 * np.pi * self.cfg.rotation_hz * (t - SPEED_WANDER / w * np.cos(w * t))

    def accel(self, start: int, n: int) -> np.ndarray:
        t = (start + np.arange(n)) / ACCEL_FS
        shaft = self.shaft_phase(t)
        amp = HARMONIC_G[:, None] * AXIS_GAIN[None, :]
        if self.cfg.unbalance:
            amp[0] *= self.cfg.unbalance_gain
        orders = np.arange(1, len(HARMONIC_G) + 1)
        angles = orders[None, :, None] * shaft[:, None, None] + self.accel_phase[None]
        g = np.sum(amp[None] * np.sin(angles), axis=1)
        g += self.rng.normal(0, ACCEL_NOISE_G, size=g.shape)
        g[:, 2] += 1.0  # gravity on z
        return np.clip(np.round(g / ACCEL_G_PER_LSB), -32768, 32767).astype("<i2")

    def audio(self, start: int, n: int) -> np.ndarray:
        t = (start + np.arange(n)) / AUDIO_FS
        shaft = self.shaft_phase(t)
        orders = self.cfg.blades * np.arange(1, len(BLADE_TONES) + 1)
        tones = BLADE_TONES @ np.sin(orders[:, None] * shaft[None, :] + self.audio_phase[:, None])
        if self.cfg.unbalance:
            tones *= 1 + 0.6 * np.sin(shaft)
        x = tones + self.rng.normal(0, AUDIO_NOISE, size=n)
        return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")

    def frame(self, stream: StreamType, timestamp_us: int, payload: bytes) -> bytes | None:
        """Encode the next frame of a stream, or None if the fault model drops it."""
        seq = self.seq[stream]
        self.seq[stream] += 1
        if self.rng.random() < self.cfg.drop_rate:
            return None
        data = encode(stream, seq, timestamp_us, payload)
        if self.rng.random() < self.cfg.corrupt_rate:
            corrupted = bytearray(data)
            corrupted[self.rng.integers(len(data))] ^= 1 << int(self.rng.integers(8))
            data = bytes(corrupted)
        return data

    def frames(self) -> Iterator[tuple[float, bytes]]:
        """Yield (seconds since start, frame bytes) in the order a real node would send them."""
        mode = "unbalance" if self.cfg.unbalance else "normal"
        hello = f"simulator {mode} at {self.cfg.rotation_hz:g} Hz, seed {self.cfg.seed}"
        yield 0.0, encode(StreamType.LOG, self.seq[StreamType.LOG], BOOT_US, hello.encode())
        self.seq[StreamType.LOG] += 1
        accel_n = audio_n = 0
        while True:
            accel_done = (accel_n + ACCEL_PER_FRAME) / ACCEL_FS
            audio_done = (audio_n + AUDIO_PER_FRAME) / AUDIO_FS
            if accel_done <= audio_done:
                ts = BOOT_US + round(accel_n * 1e6 / ACCEL_FS)
                data = self.frame(
                    StreamType.ACCEL, ts, self.accel(accel_n, ACCEL_PER_FRAME).tobytes()
                )
                accel_n += ACCEL_PER_FRAME
                due = accel_done
            else:
                ts = BOOT_US + round(audio_n * 1e6 / AUDIO_FS)
                data = self.frame(
                    StreamType.AUDIO, ts, self.audio(audio_n, AUDIO_PER_FRAME).tobytes()
                )
                audio_n += AUDIO_PER_FRAME
                due = audio_done
            if data is not None:
                yield due, data


def simulate(cfg: SimConfig, port: str, duration: float | None, fast: bool) -> None:
    """Stream frames to stdout ('-') or to a serial port or pyserial URL."""
    out = sys.stdout.buffer if port == "-" else serial.serial_for_url(port, baudrate=BAUD)
    print(f"simulating {cfg} -> {port}", file=sys.stderr)
    start = time.monotonic()
    try:
        for due, data in FakeNode(cfg).frames():
            if duration is not None and due > duration:
                break
            if not fast and (wait := start + due - time.monotonic()) > 0:
                time.sleep(wait)
            out.write(data)
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if out is not sys.stdout.buffer:
            out.close()
