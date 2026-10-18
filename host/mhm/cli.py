"""mhm command line: record from a node, run a fake one, or replay a capture through the node."""

import argparse
from pathlib import Path

from mhm import DATA_DIR, device_ml
from mhm.capture.loader import load_capture
from mhm.capture.record import CaptureName, record
from mhm.capture.simulate import SimConfig, simulate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mhm")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="record a node's UART stream to data/captures")
    cap.add_argument("--port", required=True, help="serial port, pyserial URL, or - for stdin")
    cap.add_argument("--machine", required=True)
    cap.add_argument("--condition", required=True)
    cap.add_argument("--speed", required=True, help="e.g. 1500rpm")
    cap.add_argument("--run", type=int, required=True)
    cap.add_argument("--duration", type=float, help="seconds; default is until Ctrl+C or EOF")
    cap.add_argument("--notes", default="")
    cap.add_argument("--out-dir", type=Path, default=DATA_DIR / "captures")
    cap.add_argument(
        "--send",
        action="append",
        default=[],
        metavar="COMMAND",
        help='command for the node once the port opens, e.g. "learn 60 30"; repeatable',
    )

    rep = sub.add_parser("replay", help="run a capture through the node's detector on the PC")
    rep.add_argument("capture", type=Path, help="a .bin from mhm capture")
    rep.add_argument("--learn", type=float, default=60.0, help="seconds of learning")
    rep.add_argument("--calibrate", type=float, default=30.0, help="seconds of calibration")
    rep.add_argument("--margin", type=float, default=1.2, help="threshold over calibration max")
    rep.add_argument("--start", type=float, default=0.0, help="seconds to skip before learning")

    sim = sub.add_parser("simulate", help="stream synthetic frames like a real node")
    sim.add_argument("--port", default="-", help="serial port or pyserial URL; - is stdout")
    sim.add_argument("--duration", type=float, help="seconds of signal; default is forever")
    sim.add_argument("--fast", action="store_true", help="send as fast as possible")
    sim.add_argument("--rotation-hz", type=float, default=25.0)
    sim.add_argument("--unbalance", action="store_true", help="boost the 1x shaft order")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--drop-rate", type=float, default=0.0, help="fraction of frames dropped")
    sim.add_argument("--corrupt-rate", type=float, default=0.0, help="fraction with a bit flip")

    args = parser.parse_args(argv)
    match args.command:
        case "capture":
            try:
                name = CaptureName(args.machine, args.condition, args.speed, args.run)
                record(args.port, name, args.out_dir, args.duration, args.notes, tuple(args.send))
            except (ValueError, FileExistsError) as e:
                parser.error(str(e))
        case "replay":
            capture = load_capture(args.capture)
            device_ml.build()
            for kind, stream in (("accel", capture.accel), ("audio", capture.audio)):
                samples = stream.samples[int(args.start * stream.fs) :]
                print(f"{kind}: {len(samples) / stream.fs:.1f} s after skipping {args.start:g} s")
                if len(samples) == 0:
                    continue
                r = device_ml.replay(
                    kind, samples.reshape(-1), args.learn, args.calibrate, args.margin
                )
                print(device_ml.summarize(r, offset_s=args.start))
        case "simulate":
            cfg = SimConfig(
                rotation_hz=args.rotation_hz,
                unbalance=args.unbalance,
                seed=args.seed,
                drop_rate=args.drop_rate,
                corrupt_rate=args.corrupt_rate,
            )
            simulate(cfg, args.port, args.duration, args.fast)
