"""Fetch MAFAULDA normal records plus a few fault records spread across shaft speeds."""

import argparse
from pathlib import Path
from zipfile import ZipInfo

import numpy as np

from mhm import DATA_DIR
from mhm.datasets.mafaulda import URL, rotation_hz
from mhm.datasets.remote_zip import extract, open_remote_zip, total_gb


def spread(members: list[ZipInfo], count: int) -> list[ZipInfo]:
    """Pick `count` records evenly across the speed range."""
    members = sorted(members, key=lambda i: rotation_hz(Path(i.filename)))
    picks = np.linspace(0, len(members) - 1, min(count, len(members))).round().astype(int)
    return [members[i] for i in np.unique(picks)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=int, default=49, help="normal records (49 exist)")
    parser.add_argument("--fault", default="imbalance", help="fault archive, e.g. imbalance")
    parser.add_argument(
        "--classes", nargs="+", default=["6g", "20g", "35g"], help="fault subfolders"
    )
    parser.add_argument("--per-class", type=int, default=6, help="records per fault subfolder")
    parser.add_argument("--max-gb", type=float, default=1.0, help="refuse anything larger")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "mafaulda")
    args = parser.parse_args()

    normal_zip = open_remote_zip(URL.format(name="normal"))
    fault_zip = open_remote_zip(URL.format(name=args.fault))
    normal = spread([i for i in normal_zip.infolist() if i.filename.endswith(".csv")], args.normal)
    faults = []
    for cls in args.classes:
        prefix = f"{args.fault}/{cls}/"
        found = [
            i
            for i in fault_zip.infolist()
            if i.filename.startswith(prefix) and i.filename.endswith(".csv")
        ]
        if not found:
            raise SystemExit(f"no records under {prefix}")
        faults += spread(found, args.per_class)

    size = total_gb(normal) + total_gb(faults)
    print(f"{len(normal)} normal and {len(faults)} {args.fault} records, {size:.2f} GB to download")
    if size > args.max_gb:
        raise SystemExit(f"{size:.2f} GB is over --max-gb {args.max_gb}, not downloading")
    extract(normal_zip, normal, args.out)
    extract(fault_zip, faults, args.out)


if __name__ == "__main__":
    main()
