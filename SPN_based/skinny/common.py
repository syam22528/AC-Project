from __future__ import annotations

import numpy as np

# SKINNY-64-128 uses 36 rounds.
SKINNY_ROUNDS = 36

# SKINNY-64 S-box (4-bit -> 4-bit)
SBOX = np.array([0xC, 0x6, 0x9, 0x0, 0x1, 0xA, 0x2, 0xB, 0x3, 0x8, 0x5, 0xD, 0x4, 0xE, 0x7, 0xF], dtype=np.uint8)


def _lfsr_rc(rc: int) -> int:
    """Advance 6-bit SKINNY round constant LFSR."""
    rc = ((rc << 1) ^ ((rc >> 5) & 0x01) ^ ((rc >> 4) & 0x01) ^ 0x01) & 0x3F
    return rc


def _lfsr2_word32(x: int) -> np.uint32:
    """Apply LFSR2 to packed 4-bit cells in TK2 rows 0..1."""
    y = ((x << 1) & 0xEEEEEEEE) ^ (((x >> 3) ^ (x >> 2)) & 0x11111111)
    return np.uint32(y)


def _permute_tk_words(tk0: np.uint32, tk1: np.uint32) -> tuple[np.uint32, np.uint32]:
    """Apply tweakey permutation PT on two packed 32-bit words."""
    # PT = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    x = int(tk1)
    new0 = (
        ((x & 0x0000000F) << 4)
        | ((x & 0x00F0F0F0) << 8)
        | ((x & 0x0F000000) >> 24)
        | ((x & 0x00000F00) << 16)
        | ((x & 0xF0000000) >> 12)
        | ((x & 0x000F0000) >> 8)
    )
    return np.uint32(new0), np.uint32(tk0)


def _load_u32_le(x: bytes, offset: int) -> np.uint32:
    """Load a little-endian uint32 from byte array at offset."""
    return np.uint32(
        x[offset]
        | (x[offset + 1] << 8)
        | (x[offset + 2] << 16)
        | (x[offset + 3] << 24)
    )


def generate_round_subkeys(key: bytes) -> np.ndarray:
    """Generate SKINNY-64-128 round subkeys as 32-bit row0|row1 words."""
    if len(key) != 16:
        raise ValueError("SKINNY-64-128 expects a 16-byte key")

    tk1_0 = _load_u32_le(key, 0)
    tk1_1 = _load_u32_le(key, 4)
    tk2_0 = _load_u32_le(key, 8)
    tk2_1 = _load_u32_le(key, 12)

    masks = np.zeros(SKINNY_ROUNDS, dtype=np.uint32)
    rc = 0

    for r in range(SKINNY_ROUNDS):
        rc = _lfsr_rc(rc)
        masks[r] = np.uint32(
            int(tk1_0)
            ^ int(tk2_0)
            ^ ((rc << 4) & 0xF0)
            ^ ((rc << 16) & 0x300000)
        )

        tk1_0, tk1_1 = _permute_tk_words(tk1_0, tk1_1)
        tk2_0, tk2_1 = _permute_tk_words(tk2_0, tk2_1)
        tk2_0 = _lfsr2_word32(int(tk2_0))

    return masks
