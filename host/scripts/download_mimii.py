"""Fetch a small MIMII subset (one machine type, SNR, and machine ID) without the multi-GB zip."""

import argparse
from pathlib import Path

from mhm import DATA_DIR
from mhm.datasets.mimii import URL
from mhm.datasets.remote_zip import extract, open_remote_zip, total_gb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="pump", choices=["fan", "pump", "slider", "valve"])
    parser.add_argument("--snr", default="6", choices=["-6", "0", "6"], help="dB")
    parser.add_argument("--id", default="id_00", help="machine ID folder, e.g. id_00")
    parser.add_argument("--normal", type=int, default=150, help="normal clips to fetch")
    parser.add_argument("--abnormal", type=int, default=100, help="abnormal clips to fetch")
    parser.add_argument("--max-gb", type=float, default=1.0, help="refuse anything larger")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "mimii")
    args = parser.parse_args()

    url = URL.format(snr=args.snr, machine=args.machine)
    archive = open_remote_zip(url)
    files = sorted((i for i in archive.infolist() if not i.is_dir()), key=lambda i: i.filename)
    members = []
    for label, count in [("normal", args.normal), ("abnormal", args.abnormal)]:
        prefix = f"{args.machine}/{args.id}/{label}/"
        members += [i for i in files if i.filename.startswith(prefix)][:count]

    size = total_gb(members)
    print(f"{len(members)} clips, {size:.2f} GB to download")
    if size > args.max_gb:
        raise SystemExit(f"{size:.2f} GB is over --max-gb {args.max_gb}, not downloading")
    extract(url, members, args.out / f"{args.snr}dB")


if __name__ == "__main__":
    main()
