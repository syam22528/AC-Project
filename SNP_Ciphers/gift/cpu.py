from __future__ import annotations

"""CPU implementation of GIFT-64-128 using a table-based S-box strategy.

Each round applies: SubNibbles (4-bit table lookup) → PermBits (64-bit bit
permutation) → AddRoundKey+Constant (XOR with precomputed mask).

Both single-threaded and parallel (Numba prange/OpenMP) paths are supported.
Falls back to pure-Python stubs if Numba is not installed.
"""

import os

import numpy as np

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import PBOX, SBOX, generate_round_masks
except Exception:
    from common import PBOX, SBOX, generate_round_masks

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
def _sbox_layer_table(state: np.uint64, sbox: np.ndarray) -> np.uint64:
    """Apply the GIFT 4-bit S-box to all 16 nibbles of the 64-bit state.

    Each nibble (4 bits) is extracted, substituted via the table, and
    reassembled into the output word.
    """
    out = np.uint64(0)
    for i in range(16):
        nib = np.uint64((state >> np.uint64(i * 4)) & np.uint64(0xF))
        out |= np.uint64(sbox[np.int64(nib)]) << np.uint64(i * 4)
    return out


@njit(cache=True)
def _perm_bits(state: np.uint64, pbox: np.ndarray) -> np.uint64:
    """Apply the GIFT-64 bit permutation: moves bit i of `state` to PBOX[i]."""
    out = np.uint64(0)
    for i in range(64):
        bit = (state >> np.uint64(i)) & np.uint64(1)
        out |= bit << np.uint64(pbox[i])
    return out


@njit(cache=True, parallel=True)
def _encrypt_blocks_table(states: np.ndarray, round_masks: np.ndarray, sbox: np.ndarray, pbox: np.ndarray) -> np.ndarray:
    """Encrypt an array of 64-bit GIFT-64 blocks in parallel using Numba prange.

    Each element of `states` is processed independently through 28 rounds of:
      SubNibbles → PermBits → XOR round mask.
    """
    out = np.empty_like(states)
    for i in prange(states.size):
        s = np.uint64(states[i])
        for r in range(round_masks.size):
            s = _sbox_layer_table(s, sbox)
            s = _perm_bits(s, pbox)
            s ^= np.uint64(round_masks[r])
        out[i] = s
    return out


class GiftCpuOptimized:
    """GIFT-64-128 CPU cipher using a table-based S-box strategy.

    Supports single-threaded and parallel Numba JIT paths for ECB and CTR modes.
    """

    block_size = 8

    def __init__(self, use_numba: bool = True) -> None:
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self._cached_key: bytes | None = None
        self._round_masks: np.ndarray | None = None

    @staticmethod
    def _validate_inputs(data: bytes, key: bytes) -> None:
        """Raise ValueError if data is not a multiple of 8 bytes or key is not 16 bytes."""
        if len(data) % 8 != 0:
            raise ValueError("GIFT block size is 8 bytes")
        if len(key) != 16:
            raise ValueError("GIFT-64 uses 16-byte key")

    def _get_round_masks(self, key: bytes) -> np.ndarray:
        """Return precomputed round masks for `key`, recomputing only on key change."""
        if self._cached_key != key or self._round_masks is None:
            self._round_masks = generate_round_masks(key)
            self._cached_key = bytes(key)
        return self._round_masks

    def encrypt_ecb(self, data: bytes, key: bytes, workers: int = 1) -> bytes:
        """Encrypt `data` in GIFT-64-128 ECB mode.

        Args:
            data: Plaintext bytes, must be a non-zero multiple of 8.
            key: 16-byte GIFT key.
            workers: Number of Numba CPU threads for parallel block processing.
        """
        self._validate_inputs(data, key)
        if len(data) == 0:
            return b""

        round_masks = self._get_round_masks(key)
        # Interpret plaintext as an array of big-endian 64-bit integers.
        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)

        if not self.use_numba:
            raise RuntimeError("Numba JIT is required for GIFT CPU encryption")

        cpu_count = os.cpu_count() or 1
        w = max(1, min(int(workers), cpu_count))
        old_threads = get_num_threads()
        try:
            set_num_threads(w)
            out = _encrypt_blocks_table(states, round_masks, SBOX, PBOX)
        finally:
            set_num_threads(old_threads)
        return out.astype(">u8").tobytes()

    def encrypt_ctr(self, data: bytes, key: bytes, workers: int = 1, nonce: bytes | None = None) -> bytes:
        """Encrypt `data` in GIFT-64-128 CTR mode.

        Builds counter blocks (nonce || counter), encrypts them via ECB to
        obtain a keystream, then XORs the keystream with the plaintext.
        """
        if len(data) % 8 != 0:
            raise ValueError("GIFT block size is 8 bytes")
        if len(key) != 16:
            raise ValueError("GIFT-64 uses 16-byte key")
        if len(data) == 0:
            return b""

        nblocks = len(data) // 8
        ctr_blocks = build_ctr_blocks(nblocks, 8, nonce=nonce)
        keystream = self.encrypt_ecb(ctr_blocks, key, workers=workers)
        return xor_bytes(data, keystream)
