"""Loaders for public machine anomaly datasets."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    path: Path
    abnormal: bool
