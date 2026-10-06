"""HDC prototype scorer with an LFSR item memory, kept simple enough to mirror in RTL."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np

LFSR_TAPS = 0x80200003  # x^32 + x^22 + x^2 + x + 1, maximal length
THRESHOLD_BITS = 16


def lfsr_bits(seed: int, n: int) -> np.ndarray:
    """Next n output bits of a right-shift Galois LFSR that emits its LSB before each shift."""
    if not 0 < seed < 1 << 32:
        raise ValueError("seed must be a nonzero 32-bit value")
    state = seed
    out = bytearray(n)
    for i in range(n):
        bit = state & 1
        out[i] = bit
        state >>= 1
        if bit:
            state ^= LFSR_TAPS
    return np.frombuffer(out, np.uint8).copy()


@dataclass(frozen=True)
class ItemMemory:
    ids: np.ndarray  # (n_features, dim) bits
    levels: np.ndarray  # (n_levels, dim) bits
    tiebreak: np.ndarray  # (dim,) bits


def item_memory(seed: int, dim: int, n_levels: int, n_features: int) -> ItemMemory:
    """Draw all hypervectors from one LFSR stream: tiebreak, level base, thresholds, IDs."""
    bits = lfsr_bits(seed, dim * (2 + THRESHOLD_BITS + n_features))
    tiebreak, base = bits[:dim], bits[dim : 2 * dim]
    threshold_bits = bits[2 * dim : (2 + THRESHOLD_BITS) * dim].reshape(dim, THRESHOLD_BITS)
    thresholds = threshold_bits.astype(np.int64) @ (1 << np.arange(THRESHOLD_BITS))
    ids = bits[(2 + THRESHOLD_BITS) * dim :].reshape(n_features, dim)
    # Level l flips about l / (2 (L - 1)) of the bits, so level L - 1 is orthogonal to level 0.
    cut = np.arange(n_levels) * (1 << (THRESHOLD_BITS - 1)) // (n_levels - 1)
    levels = base[None, :] ^ (thresholds[None, :] < cut[:, None])
    return ItemMemory(ids, levels.astype(np.uint8), tiebreak)


@dataclass
class HdcScorer:
    """Distance to one bundled normal prototype: Hamming (binary) or cosine (bipolar)."""

    dim: int = 4096
    n_levels: int = 16
    mode: Literal["binary", "bipolar"] = "binary"
    seed: int = 0x1234ABCD
    range_sigma: float = 6.0  # quantizer span in normal stds; narrower clips anomalies to one level
    chunk: int = 2048
    lo: np.ndarray = field(init=False, repr=False)
    width: np.ndarray = field(init=False, repr=False)
    memory: ItemMemory = field(init=False, repr=False)
    bound: np.ndarray = field(init=False, repr=False)
    prototype: np.ndarray = field(init=False, repr=False)

    def fit(self, normal_windows: np.ndarray) -> Self:
        x = np.asarray(normal_windows, np.float64)
        std = x.std(axis=0)
        self.lo = x.mean(axis=0) - self.range_sigma * std
        self.width = np.where(std > 0, 2 * self.range_sigma * std, 1.0)
        self.memory = item_memory(self.seed, self.dim, self.n_levels, x.shape[1])
        ids, levels = self.memory.ids, self.memory.levels
        match self.mode:
            case "binary":
                self.bound = ids[:, None, :] ^ levels[None, :, :]
            case "bipolar":
                # Bit b maps to (-1)^b, which turns the XOR binding into a product.
                ids_pm, levels_pm = 1 - 2 * ids.astype(np.int8), 1 - 2 * levels.astype(np.int8)
                self.bound = ids_pm[:, None, :] * levels_pm[None, :, :]
        total = sum(e.sum(axis=0, dtype=np.int64) for e in self.encoded_chunks(x))
        self.prototype = self.majority(total, len(x)) if self.mode == "binary" else total
        return self

    def quantize(self, x: np.ndarray) -> np.ndarray:
        q = np.floor((x - self.lo) / self.width * self.n_levels)
        return np.clip(q, 0, self.n_levels - 1).astype(np.intp)

    def majority(self, ones: np.ndarray, votes: int) -> np.ndarray:
        """1 where more than half the votes are 1; exact ties take the tiebreak bit."""
        return np.where(2 * ones == votes, self.memory.tiebreak, 2 * ones > votes).astype(np.uint8)

    def encoded_chunks(self, x: np.ndarray) -> Iterator[np.ndarray]:
        """Bundle each window's bound feature vectors: majority bits (binary) or sums (bipolar)."""
        for start in range(0, len(x), self.chunk):
            q = self.quantize(x[start : start + self.chunk])
            acc = np.zeros((len(q), self.dim), np.int16)
            for f in range(q.shape[1]):
                acc += self.bound[f, q[:, f]]
            yield self.majority(acc, q.shape[1]) if self.mode == "binary" else acc

    def score(self, windows: np.ndarray) -> np.ndarray:
        x = np.asarray(windows, np.float64)
        parts = []
        for e in self.encoded_chunks(x):
            if self.mode == "binary":
                parts.append(np.count_nonzero(e != self.prototype, axis=1) / self.dim)
            else:
                cosine = (e @ self.prototype) / (
                    np.linalg.norm(e, axis=1) * np.linalg.norm(self.prototype)
                )
                parts.append(1 - cosine)
        return np.concatenate(parts) if parts else np.empty(0)
