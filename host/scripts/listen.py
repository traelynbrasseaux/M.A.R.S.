"""Play the node microphone through the computer's audio output."""

import argparse
import queue
import time
from contextlib import suppress

import numpy as np
import serial
import sounddevice as sd

from mhm.capture.frame import AUDIO_FS, FrameParser, StreamType
from mhm.capture.record import BAUD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--gain", type=float, default=4.0, help="playback gain; default 4")
    parser.add_argument("--duration", type=float, help="seconds; otherwise until Ctrl+C")
    args = parser.parse_args()
    if not np.isfinite(args.gain) or args.gain < 0:
        parser.error("gain must be finite and nonnegative")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be finite and positive")

    blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    pending = np.empty(0, dtype=np.float32)

    def playback(outdata, frames, _time, _status):
        nonlocal pending
        outdata.fill(0)
        offset = 0
        while offset < frames:
            if not pending.size:
                try:
                    pending = blocks.get_nowait()
                except queue.Empty:
                    break
            count = min(frames - offset, pending.size)
            outdata[offset : offset + count, 0] = pending[:count]
            pending = pending[count:]
            offset += count

    port = serial.serial_for_url(args.port, baudrate=BAUD, timeout=0.1, do_not_open=True)
    port.dtr = port.rts = False
    frames_received = 0
    decoder = FrameParser()
    try:
        port.open()
        with sd.OutputStream(
            samplerate=int(AUDIO_FS),
            channels=1,
            dtype="float32",
            blocksize=256,
            latency="high",
            callback=playback,
        ):
            device = sd.query_devices(kind="output")
            print(f"Listening on {args.port} through {device['name']} (gain {args.gain:g}).")
            print("Use headphones or low speaker volume. Press Ctrl+C to stop.", flush=True)
            start = time.monotonic()
            while args.duration is None or time.monotonic() - start < args.duration:
                for frame in decoder.feed(port.read(port.in_waiting or 1)):
                    if frame.stream != StreamType.AUDIO:
                        continue
                    audio = np.frombuffer(frame.payload, dtype="<i2").astype(np.float32)
                    audio *= args.gain / 32768.0
                    np.clip(audio, -1, 1, out=audio)
                    if blocks.full():
                        with suppress(queue.Empty):
                            blocks.get_nowait()
                    blocks.put_nowait(audio)
                    frames_received += 1
    except KeyboardInterrupt:
        pass
    finally:
        port.close()
        print(f"Stopped. Audio frames: {frames_received}; CRC errors: {decoder.crc_errors}.")


if __name__ == "__main__":
    main()
