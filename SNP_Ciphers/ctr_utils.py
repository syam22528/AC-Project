"""Shared CTR-mode helpers used by all cipher implementations.

These utilities centralize counter block construction and XOR keystream
application so CTR logic does not live in benchmark scripts.
"""

from __future__ import annotations

import os

import numpy as np


def build_ctr_blocks(nblocks: int, block_bytes: int, nonce: bytes | None = None) -> bytes:
    """Build CTR counter blocks in nonce||counter form.

    Args:
        nblocks: number of counter blocks to produce
        block_bytes: block size in bytes for the cipher
        nonce: optional nonce prefix. If omitted, a random nonce of
            ``block_bytes // 2`` bytes is generated.

    Returns:
        Concatenated counter blocks as bytes.
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

    # Fast path for common cipher block sizes where counter fits in <= 8 bytes.
    # This avoids Python's per-block integer->bytes conversion in large benchmarks.
    if ctr_len <= 8:
        out = np.empty((nblocks, block_bytes), dtype=np.uint8)

        nonce_arr = np.frombuffer(nonce, dtype=np.uint8)
        out[:, : len(nonce)] = nonce_arr

        counters = np.arange(nblocks, dtype=np.uint64)
        be_counters = counters.astype(">u8").view(np.uint8).reshape(nblocks, 8)
        out[:, len(nonce) :] = be_counters[:, 8 - ctr_len :]
        return out.reshape(-1).tobytes()

    # Generic fallback for unusual configurations with very wide counters.
    out = bytearray(nblocks * block_bytes)
    for i in range(nblocks):
        base = i * block_bytes
        out[base : base + len(nonce)] = nonce
        out[base + len(nonce) : base + block_bytes] = i.to_bytes(ctr_len, "big", signed=False)
    return bytes(out)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two equally-sized byte strings."""
    if len(a) != len(b):
        raise ValueError("XOR inputs must have equal length")
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()
