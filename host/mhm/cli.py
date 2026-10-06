"""mhm command line: record from a node, or run a fake one."""

import argparse
from pathlib import Path

from mhm import DATA_DIR
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
                record(args.port, name, args.out_dir, args.duration, args.notes)
            except (ValueError, FileExistsError) as e:
                parser.error(str(e))
        case "simulate":
            cfg = SimConfig(
                rotation_hz=args.rotation_hz,
                unbalance=args.unbalance,
                seed=args.seed,
                drop_rate=args.drop_rate,
                corrupt_rate=args.corrupt_rate,
            )
            simulate(cfg, args.port, args.duration, args.fast)
