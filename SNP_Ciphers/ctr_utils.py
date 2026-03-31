"""Shared CTR-mode utilities used by all cipher CPU and GPU backends.

Centralises counter-block construction and keystream XOR so that CTR-mode
logic is not duplicated across cipher modules or benchmark scripts.
"""

from __future__ import annotations

import os

import numpy as np


def build_ctr_blocks(nblocks: int, block_bytes: int, nonce: bytes | None = None) -> bytes:
    """Construct `nblocks` counter blocks in nonce‖counter big-endian format.

    Each block is `block_bytes` wide.  The nonce occupies the most-significant
    bytes and the counter occupies the remaining (least-significant) bytes,
    incrementing from 0.

    Args:
        nblocks:     Number of counter blocks to generate.
        block_bytes: Cipher block size in bytes (e.g. 16 for AES, 8 for GIFT).
        nonce:       Optional nonce bytes.  Must be shorter than `block_bytes`.
                     If omitted, a random nonce of ``block_bytes // 2`` bytes
                     is generated using os.urandom.

    Returns:
        Concatenated counter blocks as a flat bytes object of length
        ``nblocks * block_bytes``.

    Raises:
        ValueError: If nonce is as long or longer than the block, if the
                    counter space is exhausted, or if arguments are invalid.
    """
    if nblocks < 0:
        raise ValueError("nblocks must be non-negative")
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")

    if nonce is None:
        nonce = os.urandom(block_bytes // 2)
    if len(nonce) >= block_bytes:
        raise ValueError("nonce must be shorter than block size")

    ctr_len = block_bytes - len(nonce)
    max_blocks = 1 << (8 * ctr_len)
    if nblocks > max_blocks:
        raise ValueError("counter space exhausted for provided nonce and block size")

    if nblocks == 0:
        return b""

    # Fast path for block sizes where the counter fits in ≤ 8 bytes.
    # Avoids Python's per-block integer-to-bytes conversion for large benchmarks
    # by using NumPy vectorised operations instead.
    if ctr_len <= 8:
        out = np.empty((nblocks, block_bytes), dtype=np.uint8)

        nonce_arr = np.frombuffer(nonce, dtype=np.uint8)
        out[:, : len(nonce)] = nonce_arr

        # Generate counter values as big-endian uint64, then slice the needed bytes.
        counters = np.arange(nblocks, dtype=np.uint64)
        be_counters = counters.astype(">u8").view(np.uint8).reshape(nblocks, 8)
        out[:, len(nonce) :] = be_counters[:, 8 - ctr_len :]
        return out.reshape(-1).tobytes()

    # Generic fallback for unusual block sizes with very wide counter fields.
    out = bytearray(nblocks * block_bytes)
    for i in range(nblocks):
        base = i * block_bytes
        out[base : base + len(nonce)] = nonce
        out[base + len(nonce) : base + block_bytes] = i.to_bytes(ctr_len, "big", signed=False)
    return bytes(out)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two equal-length byte strings element-wise using NumPy.

    Used to apply a keystream to plaintext (or vice versa) in CTR mode.

    Args:
        a: First byte string.
        b: Second byte string, must be the same length as `a`.

    Returns:
        Element-wise XOR result as bytes.

    Raises:
        ValueError: If `a` and `b` differ in length.
    """
    if len(a) != len(b):
        raise ValueError("XOR inputs must have equal length")
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()
