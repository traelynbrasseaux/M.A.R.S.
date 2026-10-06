"""MIMII: 10 s, 16 kHz, 8-channel 16-bit WAV clips of industrial machines (Purohit et al., 2019)."""

import wave
from pathlib import Path

import numpy as np

from mhm.datasets import Clip

URL = "https://zenodo.org/records/3384388/files/{snr}_dB_{machine}.zip?download=1"


def list_clips(root: Path, machine: str = "pump", machine_id: str = "id_00") -> list[Clip]:
    """Clips under root/{machine}/{machine_id}/{normal,abnormal}/, normal first."""
    base = root / machine / machine_id
    return [
        Clip(path, label == "abnormal")
        for label in ("normal", "abnormal")
        for path in sorted((base / label).glob("*.wav"))
    ]


def load_wav(path: Path, channel: int = 0) -> tuple[np.ndarray, int]:
    """One channel of a 16-bit PCM WAV as int16, plus its sample rate."""
    with path.open("rb") as f, wave.open(f) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, "<i2").reshape(-1, w.getnchannels())
        return data[:, channel].astype(np.int16), w.getframerate()
