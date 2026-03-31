"""PRESENT-128 cipher constants and key schedule."""

from __future__ import annotations

import numpy as np

# PRESENT 4-bit S-box (16 entries)
SBOX = np.array(
    [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],
    dtype=np.uint8,
)

# PRESENT 64-bit P-box (bit permutation indices)
PBOX = np.array([(16 * i) % 63 if i != 63 else 63 for i in range(64)], dtype=np.uint8)


def _rotl(value: int, shift: int, width: int) -> int:
    """Rotate left operation on arbitrary-width integers."""
    mask = (1 << width) - 1
    shift %= width
    return ((value << shift) & mask) | ((value & mask) >> (width - shift))


def generate_round_keys(key: bytes) -> np.ndarray:
    """Generate 32 round keys from a 128-bit PRESENT key."""
    if len(key) != 16:
        raise ValueError("PRESENT-128 requires a 16-byte key")

    k = int.from_bytes(key, "big")
    round_keys = np.zeros(32, dtype=np.uint64)
    mask = (1 << 128) - 1

    for rnd in range(1, 33):
        round_keys[rnd - 1] = (k >> 64) & 0xFFFFFFFFFFFFFFFF
        if rnd == 32:
            break
        k = _rotl(k, 61, 128) & mask
        top0 = (k >> 124) & 0xF
        top1 = (k >> 120) & 0xF
        k &= (1 << 120) - 1
        k |= int(SBOX[top1]) << 120
        k |= int(SBOX[top0]) << 124
        k ^= (rnd & 0x1F) << 62

    return round_keys
