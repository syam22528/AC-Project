from __future__ import annotations

"""SKINNY-64-128 shared constants and key schedule.

Provides the S-box, round count, and the function that derives the 36
per-round subkeys (as 32-bit words) from a 128-bit tweakey using the
SKINNY-64-128 tweakey schedule (TK1 and TK2 halves with PT permutation
and TK2 LFSR update).
"""

import numpy as np

# SKINNY-64-128 uses 36 encryption rounds.
SKINNY_ROUNDS = 36

# SKINNY-64 4-bit S-box: maps each nibble value 0–15 to its substituted output.
SBOX = np.array([0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF], dtype=np.uint8)


def _lfsr_rc(rc: int) -> int:
    """Advance the 6-bit SKINNY round constant LFSR by one step.

    The LFSR polynomial is x^6 + x^4 + 1 evaluated with an additional
    '1' injection, producing the sequence of round constants for SKINNY.
    """
    rc = ((rc << 1) ^ ((rc >> 5) & 0x01) ^ ((rc >> 4) & 0x01) ^ 0x01) & 0x3F
    return rc


def _lfsr2_word32(x: int) -> np.uint32:
    """Apply the TK2 LFSR (LFSR-2) to a packed 32-bit word of 4-bit cells.

    Each 4-bit nibble is updated independently: the new nibble is a
    left-shift of the original with feedback from bits 3 and 2 driving bit 0.
    This corresponds to the SKINNY TK2 update rule.
    """
    y = ((x << 1) & 0xEEEEEEEE) ^ (((x >> 3) ^ (x >> 2)) & 0x11111111)
    return np.uint32(y)


def _permute_tk_words(tk0: np.uint32, tk1: np.uint32) -> tuple[np.uint32, np.uint32]:
    """Apply the SKINNY tweakey cell permutation PT to two packed 32-bit TK words.

    PT = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7].
    The 16-cell TK register is packed as two 32-bit words (tk0 holds cells 0-7,
    tk1 holds cells 8-15, each cell being a 4-bit nibble).
    The permuted new tk0 is derived from the old tk1; new tk1 becomes the old tk0.
    """
    x = int(tk1)
    new0 = (
        ((x & 0x0000000F) << 4)
        | ((x & 0x00F0F0F0) << 8)
        | ((x & 0x0F000000) >> 24)
        | ((x & 0x00000F00) << 16)
        | ((x & 0xF0000000) >> 12)
        | ((x & 0x000F0000) >> 8)
    )
    # New tk1 is simply the previous tk0 (cells 0-7 shift to the upper half).
    return np.uint32(new0), np.uint32(tk0)


def _load_u32_le(x: bytes, offset: int) -> np.uint32:
    """Load a little-endian 32-bit unsigned integer from byte array `x` at `offset`."""
    return np.uint32(
        x[offset]
        | (x[offset + 1] << 8)
        | (x[offset + 2] << 16)
        | (x[offset + 3] << 24)
    )


def generate_round_subkeys(key: bytes) -> np.ndarray:
    """Generate 36 SKINNY-64-128 round subkeys as 32-bit XOR masks.

    The 128-bit tweakey is split into two 64-bit halves: TK1 (lower 8 bytes)
    and TK2 (upper 8 bytes).  Each 64-bit half is further split into two
    packed 32-bit words holding nibble-level cells.

    Each round:
      1. XORs the upper two rows of TK1 and TK2 with the 6-bit round constant.
      2. Applies PT to permute cells within each TK half.
      3. Updates TK2 with the nibble-level LFSR-2.

    The returned uint32 array has length 36; element r is XORed into the
    lower 32 bits of the 64-bit cipher state at round r.
    """
    if len(key) != 16:
        raise ValueError("SKINNY-64-128 expects a 16-byte key")

    # Load each 64-bit tweakey half as two little-endian 32-bit words.
    tk1_0 = _load_u32_le(key, 0)
    tk1_1 = _load_u32_le(key, 4)
    tk2_0 = _load_u32_le(key, 8)
    tk2_1 = _load_u32_le(key, 12)

    masks = np.zeros(SKINNY_ROUNDS, dtype=np.uint32)
    rc = 0

    for r in range(SKINNY_ROUNDS):
        # Advance LFSR to get the current round constant.
        rc = _lfsr_rc(rc)
        # Round subkey = XOR of TK1 and TK2 row-0 words plus round-constant bits.
        masks[r] = np.uint32(
            int(tk1_0)
            ^ int(tk2_0)
            ^ ((rc << 4) & 0xF0)          # bits [5:4] of rc into nibbles of column 0
            ^ ((rc << 16) & 0x300000)     # bits [1:0] of rc into row 1, nibble 0
        )

        # Apply the PT permutation and LFSR-2 to advance the tweakey state.
        tk1_0, tk1_1 = _permute_tk_words(tk1_0, tk1_1)
        tk2_0, tk2_1 = _permute_tk_words(tk2_0, tk2_1)
        tk2_0 = _lfsr2_word32(int(tk2_0))

    return masks
