"""PRESENT-128 cipher constants and key schedule.

Provides the S-box, P-box, and the generate_round_keys function that
derives the 32 round keys from a 128-bit key using the PRESENT-128 key
schedule (61-bit left rotation, LFSR-driven top-nibble substitution,
and mixing of the 5-bit round counter).
"""

from __future__ import annotations

import numpy as np

# PRESENT-128 4-bit S-box: maps each nibble input 0–15 to its substituted output.
SBOX = np.array(
    [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],
    dtype=np.uint8,
)

# PRESENT-128 P-box: PBOX[i] = (16 * i) mod 63 for i < 63; bit 63 maps to itself.
PBOX = np.array([(16 * i) % 63 if i != 63 else 63 for i in range(64)], dtype=np.uint8)


def _rotl(value: int, shift: int, width: int) -> int:
    """Rotate `value` left by `shift` bits within a `width`-bit word."""
    mask = (1 << width) - 1
    shift %= width
    return ((value << shift) & mask) | ((value & mask) >> (width - shift))


def generate_round_keys(key: bytes) -> np.ndarray:
    """Generate 32 round keys from a 128-bit PRESENT key.

    The PRESENT-128 key schedule maintains a 128-bit key register and
    extracts the upper 64 bits as each round key.  After each extraction
    (except the last) it updates the register by:
      1. Rotating the 128-bit register left by 61 positions.
      2. Substituting the top two nibbles through the S-box.
      3. XORing bits [66:62] with the 5-bit round counter.

    Returns a uint64 array of length 32.
    """
    if len(key) != 16:
        raise ValueError("PRESENT-128 requires a 16-byte key")

    k = int.from_bytes(key, "big")
    round_keys = np.zeros(32, dtype=np.uint64)
    mask = (1 << 128) - 1

    for rnd in range(1, 33):
        # Extract the upper 64 bits as this round's key.
        round_keys[rnd - 1] = (k >> 64) & 0xFFFFFFFFFFFFFFFF
        if rnd == 32:
            break
        # Step 1: rotate the 128-bit register left by 61 bits.
        k = _rotl(k, 61, 128) & mask
        # Step 2: substitute the top two nibbles (bits [127:124] and [123:120]).
        top0 = (k >> 124) & 0xF
        top1 = (k >> 120) & 0xF
        k &= (1 << 120) - 1
        k |= int(SBOX[top1]) << 120
        k |= int(SBOX[top0]) << 124
        # Step 3: XOR the 5-bit round counter into bits [66:62].
        k ^= (rnd & 0x1F) << 62

    return round_keys
