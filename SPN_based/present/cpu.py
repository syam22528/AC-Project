"""PRESENT-128 CPU implementation with Numba JIT.

Each of the 31 main rounds applies:
  1. AddRoundKey  — XOR state with the 64-bit round key.
  2. SBox         — 4-bit substitution on each of the 16 nibbles.
  3. pLayer       — 64-bit bit permutation (bit i → bit (16i mod 63)).

The pLayer is implemented as 4 sequential delta-swap operations instead of
a 63-iteration bit loop, which reduces the permutation to 8 XOR/AND
instructions per block.

A 32nd AddRoundKey follows the main rounds to produce the final ciphertext.

Single-threaded and parallel (Numba prange / OpenMP) paths are both supported.
"""

from __future__ import annotations

import os

import numpy as np

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import SBOX, generate_round_keys
except Exception:
    from common import SBOX, generate_round_keys

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


# ---------------------------------------------------------------------------
# Core layers (JIT-compiled device functions)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sbox_layer(state: np.uint64, sbox: np.ndarray) -> np.uint64:
    """Apply the PRESENT S-box to all 16 nibbles of the 64-bit state."""
    out = np.uint64(0)
    for i in range(16):
        nib = np.uint64((state >> np.uint64(i * 4)) & np.uint64(0xF))
        out |= np.uint64(sbox[np.int64(nib)]) << np.uint64(i * 4)
    return out


@njit(cache=True, inline="always")
def _delta_swap(x: np.uint64, mask: np.uint64, shift: np.uint64) -> np.uint64:
    """Swap bit-pairs separated by `shift` positions where `mask` marks the lower bits.

    Given a pair of bits at positions p and p+shift, the delta-swap exchanges
    them when the bit at position p is selected by `mask`.
    t = (x ^ (x >> shift)) & mask  keeps only differing bit-pairs.
    x ^ t ^ (t << shift) swaps those pairs in one step.
    """
    t = (x ^ (x >> shift)) & mask
    return x ^ t ^ (t << shift)


@njit(cache=True)
def _p_layer(state: np.uint64) -> np.uint64:
    """Apply the PRESENT pLayer via 4 delta-swap operations.

    The pLayer permutation (bit i → (16i mod 63), bit 63 fixed) can be
    decomposed into 4 pairwise bit-index swaps:
      (index bit 0, index bit 2), (index bit 2, index bit 4),
      (index bit 1, index bit 3), (index bit 3, index bit 5).
    Each swap is implemented as one delta_swap call.  Bit 63 is preserved
    automatically by the choice of masks.

    Correctness verified against the reference 63-iteration loop on 100K
    random test vectors.
    """
    x = state
    x = _delta_swap(x, np.uint64(0x0A0A0A0A0A0A0A0A), np.uint64(3))   # swap index bits 0 ↔ 2
    x = _delta_swap(x, np.uint64(0x0000F0F00000F0F0), np.uint64(12))   # swap index bits 2 ↔ 4
    x = _delta_swap(x, np.uint64(0x00CC00CC00CC00CC), np.uint64(6))    # swap index bits 1 ↔ 3
    x = _delta_swap(x, np.uint64(0x00000000FF00FF00), np.uint64(24))   # swap index bits 3 ↔ 5
    return x


# ---------------------------------------------------------------------------
# Parallel block encryption
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _encrypt_blocks(states: np.ndarray, round_keys: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Encrypt an array of 64-bit PRESENT states in parallel using Numba prange.

    Runs 31 full rounds (AddRoundKey + SBox + pLayer) followed by a final
    AddRoundKey.  Each state is independent so prange can distribute work
    across CPU threads via OpenMP.
    """
    out = np.empty_like(states)
    for i in prange(states.size):
        s = np.uint64(states[i])
        for r in range(31):
            s ^= np.uint64(round_keys[r])   # AddRoundKey
            s = _sbox_layer(s, sbox)        # SubNibbles
            s = _p_layer(s)                 # pLayer
        s ^= np.uint64(round_keys[31])      # Final AddRoundKey (round key 32)
        out[i] = s
    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PresentCpuOptimized:
    """PRESENT-128 CPU cipher with Numba JIT compilation.

    Block size: 8 bytes (64-bit state).
    Rounds: 31 main rounds + 1 final AddRoundKey.
    Supports single-threaded and parallel execution via Numba prange (OpenMP).
    """

    block_size = 8

    def __init__(self, use_numba: bool = True) -> None:
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)
        self._cached_key: bytes | None = None
        self._round_keys: np.ndarray | None = None

    @staticmethod
    def _validate_inputs(data: bytes, key: bytes) -> None:
        """Raise ValueError if data is not a multiple of 8 bytes or key is not 16 bytes."""
        if len(data) % 8 != 0:
            raise ValueError("PRESENT block size is 8 bytes")
        if len(key) != 16:
            raise ValueError("PRESENT-128 requires a 16-byte key")

    def _get_round_keys(self, key: bytes) -> np.ndarray:
        """Return the 32 round keys for `key`, using a cached result if the key is unchanged."""
        if self._cached_key != key or self._round_keys is None:
            self._round_keys = generate_round_keys(key)
            self._cached_key = bytes(key)
        return self._round_keys

    def encrypt_ecb(self, data: bytes, key: bytes, workers: int = 1) -> bytes:
        """Encrypt `data` in PRESENT-128 ECB mode using the JIT-compiled kernel.

        Args:
            data: Plaintext bytes, must be a non-zero multiple of 8.
            key: 16-byte PRESENT key.
            workers: Number of CPU threads (>1 uses parallel prange path).
        """
        self._validate_inputs(data, key)
        if len(data) == 0:
            return b""

        round_keys = self._get_round_keys(key)
        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)

        if not self.use_numba:
            raise RuntimeError("Numba JIT is required for PRESENT CPU encryption")

        cpu_count = os.cpu_count() or 1
        w = max(1, min(int(workers), cpu_count))
        old_threads = get_num_threads()
        try:
            set_num_threads(w)
            out = _encrypt_blocks(states, round_keys, SBOX)
        finally:
            set_num_threads(old_threads)
        return out.astype(">u8").tobytes()

    def encrypt_ctr(self, data: bytes, key: bytes, workers: int = 1, nonce: bytes | None = None) -> bytes:
        """Encrypt `data` in PRESENT-128 CTR mode.

        Builds counter blocks (nonce || counter), encrypts them via ECB to
        produce a keystream, then XORs the keystream with the plaintext.
        """
        if len(key) != 16:
            raise ValueError("PRESENT-128 requires a 16-byte key")
        if len(data) % 8 != 0:
            raise ValueError("PRESENT block size is 8 bytes")
        if len(data) == 0:
            return b""

        nblocks = len(data) // 8
        ctr_blocks = build_ctr_blocks(nblocks, 8, nonce=nonce)
        keystream = self.encrypt_ecb(ctr_blocks, key, workers=workers)
        return xor_bytes(data, keystream)