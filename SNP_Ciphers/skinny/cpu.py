from __future__ import annotations

"""CPU implementation of SKINNY-64-128 with uniform table-based S-box path.

Uses Numba JIT for fast parallel block processing with a pure-Python fallback.
"""

import os

import numpy as np

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import SBOX, generate_round_subkeys
except Exception:
    from common import SBOX, generate_round_subkeys

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

    def prange(*args):  # type: ignore
        return range(*args)

    def set_num_threads(n: int) -> None:  # type: ignore
        return None

    def get_num_threads() -> int:  # type: ignore
        return 1


@njit(cache=True)
def _ror16(x: np.uint16, c: int) -> np.uint16:
    """Rotate a 16-bit value right by c bits."""
    return np.uint16((x >> np.uint16(c)) | (x << np.uint16(16 - c)))


@njit(cache=True)
def _sbox_table(state: np.uint64, sbox: np.ndarray) -> np.uint64:
    """Apply table S-box to all 16 nibbles in the 64-bit state."""
    out = np.uint64(0)
    for i in range(16):
        nib = np.uint8((state >> np.uint64(4 * i)) & np.uint64(0xF))
        out |= np.uint64(sbox[np.int64(nib)]) << np.uint64(4 * i)
    return out


@njit(cache=True)
def _round_linear(state: np.uint64) -> np.uint64:
    """Apply SKINNY ShiftRows + MixColumns linear layer."""
    row0 = np.uint16(state & np.uint64(0xFFFF))
    row1 = np.uint16((state >> np.uint64(16)) & np.uint64(0xFFFF))
    row2 = np.uint16((state >> np.uint64(32)) & np.uint64(0xFFFF))
    row3 = np.uint16((state >> np.uint64(48)) & np.uint64(0xFFFF))

    row1 = _ror16(row1, 4)
    row2 = _ror16(row2, 8)
    row3 = _ror16(row3, 12)

    row1x = np.uint16(row1 ^ row2)
    row2x = np.uint16(row2 ^ row0)
    temp = np.uint16(row3 ^ row2x)

    row3 = row2x
    row2 = row1x
    row1 = row0
    row0 = temp

    return (
        np.uint64(row0)
        | (np.uint64(row1) << np.uint64(16))
        | (np.uint64(row2) << np.uint64(32))
        | (np.uint64(row3) << np.uint64(48))
    )


@njit(cache=True, parallel=True)
def _encrypt_blocks_table(states: np.ndarray, rkeys: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Parallel ECB block encryption using table S-box path."""
    out = np.empty_like(states)
    for i in prange(states.size):
        s = np.uint64(states[i])
        for r in range(rkeys.size):
            s = _sbox_table(s, sbox)
            lo = np.uint32(s & np.uint64(0xFFFFFFFF))
            lo ^= np.uint32(rkeys[r])
            s = (s & np.uint64(0xFFFFFFFF00000000)) | np.uint64(lo)
            s ^= np.uint64(0x0000002000000000)
            s = _round_linear(s)
        out[i] = s
    return out


class SkinnyCpuOptimized:
    """SKINNY-64-128 CPU with uniform table-based S-box implementation."""

    block_size = 8

    def __init__(self, use_numba: bool = True) -> None:
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self._cached_key: bytes | None = None
        self._rkeys: np.ndarray | None = None

    @staticmethod
    def _validate_inputs(data: bytes, key: bytes) -> None:
        """Validate key length and input block alignment."""
        if len(data) % 8 != 0:
            raise ValueError("SKINNY-64 block size is 8 bytes")
        if len(key) != 16:
            raise ValueError("SKINNY-64-128 expects a 16-byte key")

    def _get_rkeys(self, key: bytes) -> np.ndarray:
        """Compute and cache round subkeys for the current key."""
        if self._cached_key != key or self._rkeys is None:
            self._rkeys = generate_round_subkeys(key)
            self._cached_key = bytes(key)
        return self._rkeys

    def encrypt_ecb(self, data: bytes, key: bytes, workers: int = 1) -> bytes:
        """Encrypt ECB blocks with configured variant.

Args:
    data: plaintext bytes, multiple of 8
    key: 16-byte SKINNY key
    workers: CPU threads for Numba parallel path
"""
        self._validate_inputs(data, key)
        if len(data) == 0:
            return b""

        rkeys = self._get_rkeys(key)
        states = np.frombuffer(data, dtype=np.uint64).astype(np.uint64)

        if not self.use_numba:
            raise RuntimeError("Numba JIT is required for SKINNY CPU encryption")

        cpu_count = os.cpu_count() or 1
        w = max(1, min(int(workers), cpu_count))
        old_threads = get_num_threads()
        try:
            set_num_threads(w)
            out = _encrypt_blocks_table(states, rkeys, SBOX)
        finally:
            set_num_threads(old_threads)
        return out.astype(np.uint64).tobytes()

    def encrypt_ctr(self, data: bytes, key: bytes, workers: int = 1, nonce: bytes | None = None) -> bytes:
        """Encrypt data in CTR mode using ECB(counter) keystream generation."""
        if len(data) % 8 != 0:
            raise ValueError("SKINNY-64 block size is 8 bytes")
        if len(key) != 16:
            raise ValueError("SKINNY-64-128 expects a 16-byte key")
        if len(data) == 0:
            return b""

        nblocks = len(data) // 8
        ctr_blocks = build_ctr_blocks(nblocks, 8, nonce=nonce)
        keystream = self.encrypt_ecb(ctr_blocks, key, workers=workers)
        return xor_bytes(data, keystream)
