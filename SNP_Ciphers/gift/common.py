from __future__ import annotations

"""GIFT-64-128 shared constants and key schedule.

Provides the S-box, permutation box, round constants, and the function
that derives the 28 per-round XOR masks (combined AddRoundKey + round
constant) used by both the CPU and GPU backends.
"""

import numpy as np

# Number of encryption rounds for GIFT-64-128.
GIFT_ROUNDS = 28

# GIFT-64 4-bit S-box: maps each nibble value 0–15 to its substituted output.
SBOX = np.array([1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14], dtype=np.uint8)

# GIFT-64 bit permutation: PBOX[i] is the output position for bit i of the state.
PBOX = np.array(
    [
        0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3,
        4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7,
        8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
        12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15,
    ],
    dtype=np.uint8,
)

# Pre-defined GIFT round constants (6-bit LFSR sequence, 62 entries).
ROUND_CONSTANTS = np.array(
    [
        0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
        0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
        0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
        0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
        0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
        0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28,
        0x10, 0x20,
    ],
    dtype=np.uint8,
)


def _key_nibbles_from_bytes(key: bytes) -> np.ndarray:
    """Unpack a 16-byte GIFT key into 32 little-endian 4-bit nibble cells.

    Cell i holds bits [4i+3 : 4i] of the big-endian integer form of the key.
    """
    if len(key) != 16:
        raise ValueError("GIFT-64 uses a 128-bit key (16 bytes)")
    k = int.from_bytes(key, "big")
    out = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        out[i] = np.uint8((k >> (4 * i)) & 0xF)
    return out


def _update_key_state(key_nibs: np.ndarray) -> np.ndarray:
    """Advance the GIFT key state by one round using the GIFT key schedule.

    Rotates the 32-nibble key register by 8 positions, then applies the
    nibble-level rotations to k0 (>> 12 within 4-nibble group) and k1
    (>> 2 within 4-nibble group with wrap-around).
    """
    temp = np.empty_like(key_nibs)
    for i in range(32):
        temp[i] = key_nibs[(i + 8) % 32]

    out = np.empty_like(key_nibs)
    out[:24] = temp[:24]

    # k0: rotate the top nibble group right by 1 (equivalent to >> 12 on that word).
    out[24] = temp[27]
    out[25] = temp[24]
    out[26] = temp[25]
    out[27] = temp[26]

    # k1: rotate 2 bits right across 4 consecutive nibble cells.
    out[28] = np.uint8(((int(temp[28]) & 0xC) >> 2) | ((int(temp[29]) & 0x3) << 2))
    out[29] = np.uint8(((int(temp[29]) & 0xC) >> 2) | ((int(temp[30]) & 0x3) << 2))
    out[30] = np.uint8(((int(temp[30]) & 0xC) >> 2) | ((int(temp[31]) & 0x3) << 2))
    out[31] = np.uint8(((int(temp[31]) & 0xC) >> 2) | ((int(temp[28]) & 0x3) << 2))
    return out


def _build_round_mask(key_nibs: np.ndarray, rc: int) -> np.uint64:
    """Build the combined AddRoundKey + round-constant XOR mask for one round.

    The 64-bit mask is XORed with the state after SubNibbles + PermBits.
    It encodes:
      - Key bits from k[0..15] at bit positions 4i (every 4th bit)
      - Key bits from k[16..31] at bit positions 4i+1
      - 6 round-constant bits at bits 3, 7, 11, 15, 19, 23
      - A fixed '1' at bit 63 (domain separator)
    """
    key_bits = np.zeros(128, dtype=np.uint8)
    for i in range(32):
        nib = int(key_nibs[i])
        key_bits[4 * i + 0] = np.uint8((nib >> 0) & 1)
        key_bits[4 * i + 1] = np.uint8((nib >> 1) & 1)
        key_bits[4 * i + 2] = np.uint8((nib >> 2) & 1)
        key_bits[4 * i + 3] = np.uint8((nib >> 3) & 1)

    mask = 0

    # Inject key bits k[0..15] and k[16..31] in the lower half of the state.
    for i in range(16):
        mask ^= int(key_bits[i]) << (4 * i)
        mask ^= int(key_bits[i + 16]) << (4 * i + 1)

    # Inject the 6-bit round constant into bits 3, 7, 11, 15, 19, 23.
    mask ^= ((rc >> 0) & 1) << 3
    mask ^= ((rc >> 1) & 1) << 7
    mask ^= ((rc >> 2) & 1) << 11
    mask ^= ((rc >> 3) & 1) << 15
    mask ^= ((rc >> 4) & 1) << 19
    mask ^= ((rc >> 5) & 1) << 23
    # Fixed '1' at bit 63 acts as a domain-separation flag.
    mask ^= 1 << 63

    return np.uint64(mask)


def generate_round_masks(key: bytes) -> np.ndarray:
    """Generate the 28 combined AddRoundKey + round-constant masks for GIFT-64-128.

    Returns a uint64 array of length GIFT_ROUNDS; element r is XORed with
    the state after the SubNibbles and PermBits steps of round r.
    """
    key_nibs = _key_nibbles_from_bytes(key)
    masks = np.zeros(GIFT_ROUNDS, dtype=np.uint64)

    for r in range(GIFT_ROUNDS):
        masks[r] = _build_round_mask(key_nibs, int(ROUND_CONSTANTS[r]))
        key_nibs = _update_key_state(key_nibs)

    return masks
