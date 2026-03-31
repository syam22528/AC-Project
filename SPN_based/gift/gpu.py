from __future__ import annotations

"""GPU implementation of GIFT-64-128 using Numba CUDA.

Provides table and bitsliced S-box variants for benchmark comparisons.

Optimizations applied:
- Combined S-box + P-box nibble scatter tables (table variant)
  replaces 80 loop iterations (16 S-box + 64 P-box) with 16 table lookups.
- Precomputed P-box nibble scatter table (bitsliced variant)
  replaces 64-iteration bit-by-bit loop with 16 table lookups.
- Thread coarsening: 4 blocks per thread (was 2).
- Shared memory for round keys.
"""

import time
from dataclasses import dataclass

import numpy as np
import numba
from numba import cuda

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import PBOX, SBOX, generate_round_masks
except Exception:
    from common import PBOX, SBOX, generate_round_masks


# ---------------------------------------------------------------------------
# Precomputed scatter tables (host-side, uploaded to constant memory)
# ---------------------------------------------------------------------------

def _compute_pbox_scatter() -> np.ndarray:
    """P-box nibble scatter: 16 nibble positions × 16 values = 256 entries.

    PBOX_SCATTER[k*16 + v] gives the 64-bit word with the 4 bits of
    nibble value `v` scattered to their correct output positions for
    input nibble `k`.
    """
    table = np.zeros(256, dtype=np.uint64)
    for k in range(16):
        for v in range(16):
            val = np.uint64(0)
            for b in range(4):
                if v & (1 << b):
                    val |= np.uint64(1) << np.uint64(int(PBOX[k * 4 + b]))
            table[k * 16 + v] = val
    return table


def _compute_sp_scatter() -> np.ndarray:
    """Combined S-box + P-box scatter: 16 × 16 = 256 entries.

    SP_SCATTER[k*16 + v] = pbox_scatter(k, SBOX[v])
    Fuses the SubNibbles + PermBits layers into a single table lookup.
    """
    table = np.zeros(256, dtype=np.uint64)
    for k in range(16):
        for v in range(16):
            sv = int(SBOX[v])
            val = np.uint64(0)
            for b in range(4):
                if sv & (1 << b):
                    val |= np.uint64(1) << np.uint64(int(PBOX[k * 4 + b]))
            table[k * 16 + v] = val
    return table


PBOX_SCATTER_CONST = _compute_pbox_scatter()
SP_SCATTER_CONST = _compute_sp_scatter()

# Thread coarsening factor: each CUDA thread processes this many blocks.
BLOCKS_PER_THREAD = 4


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def sp_layer_table_dev(x):
    """Combined S-box + P-box via scatter table (constant memory)."""
    sp = cuda.const.array_like(SP_SCATTER_CONST)
    out = numba.uint64(0)
    for k in range(16):
        nib = (x >> numba.uint64(k * 4)) & numba.uint64(0xF)
        out |= sp[k * 16 + nib]
    return out


@cuda.jit(device=True)
def sbox_layer_bitsliced_dev(x):
    """Apply GIFT S-box via bitsliced boolean equations."""
    m = numba.uint64(0x1111111111111111)
    one = m

    x0 = x & m
    x1 = (x >> 1) & m
    x2 = (x >> 2) & m
    x3 = (x >> 3) & m

    y0 = one ^ x0 ^ x1 ^ (x0 & x1) ^ x2 ^ x3
    y1 = x0 ^ (x0 & x1) ^ x2 ^ (x0 & x2) ^ x3
    y2 = x1 ^ x2 ^ (x0 & x3) ^ (x1 & x3) ^ (x1 & x2 & x3)
    y3 = x0 ^ (x1 & x3) ^ (x0 & x2 & x3)

    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)


@cuda.jit(device=True)
def p_layer_scatter_dev(x):
    """P-box via nibble scatter table (constant memory)."""
    pt = cuda.const.array_like(PBOX_SCATTER_CONST)
    out = numba.uint64(0)
    for k in range(16):
        nib = (x >> numba.uint64(k * 4)) & numba.uint64(0xF)
        out |= pt[k * 16 + nib]
    return out


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@cuda.jit
def gift_encrypt_kernel_table(input_data, output_data, rkeys, nblocks):
    """GPU kernel: combined SP scatter table, 4 blocks/thread."""
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD
    shared_rkeys = cuda.shared.array(28, dtype=numba.uint64)

    for i in range(cuda.threadIdx.x, 28, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            continue

        s = input_data[b]
        for r in range(28):
            s = sp_layer_table_dev(s)
            s ^= shared_rkeys[r]
        output_data[b] = s


@cuda.jit
def gift_encrypt_kernel_bitsliced(input_data, output_data, rkeys, nblocks):
    """GPU kernel: bitsliced S-box + scatter P-box, 4 blocks/thread."""
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD
    shared_rkeys = cuda.shared.array(28, dtype=numba.uint64)

    for i in range(cuda.threadIdx.x, 28, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            continue

        s = input_data[b]
        for r in range(28):
            s = sbox_layer_bitsliced_dev(s)
            s = p_layer_scatter_dev(s)
            s ^= shared_rkeys[r]
        output_data[b] = s


# ---------------------------------------------------------------------------
# Timing / wrapper
# ---------------------------------------------------------------------------

@dataclass
class GpuTiming:
    """Timing breakdown for one GPU ECB encryption call."""
    total_seconds: float
    kernel_seconds: float
    h2d_d2h_seconds: float


class GiftGpuOptimized:
    """GIFT-64-128 GPU implementation with default and evaluation variants."""

    DEFAULT_VARIANT = "table"
    EVALUATION_VARIANTS = ("table", "bitsliced")

    def __init__(self, block_size: int = 256, variant: str = DEFAULT_VARIANT) -> None:
        self.block_size = int(block_size)
        if variant not in ("table", "bitsliced"):
            raise ValueError("variant must be 'table' or 'bitsliced'")
        self.variant = variant
        self._is_key_set = False

    def set_key(self, key: bytes) -> None:
        """Set 128-bit key and upload precomputed round masks to the device."""
        round_masks = generate_round_masks(key).astype(np.uint64)
        self.rkeys_device = cuda.to_device(round_masks)
        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Validate block alignment and return number of 8-byte blocks."""
        if len(data) % 8 != 0:
            raise ValueError("GIFT block size is 8 bytes")
        return len(data) // 8

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt ECB data on GPU and return ciphertext plus timing metrics."""
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)

        t0 = time.perf_counter()

        h2d_t0 = time.perf_counter()
        d_in = cuda.to_device(states)
        d_out = cuda.device_array_like(d_in)
        h2d_seconds = time.perf_counter() - h2d_t0

        logical_threads = (nblocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
        grid = ((logical_threads + self.block_size - 1) // self.block_size,)

        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record()

        if self.variant == "bitsliced":
            gift_encrypt_kernel_bitsliced[grid[0], self.block_size](
                d_in, d_out, self.rkeys_device, np.int32(nblocks)
            )
        else:
            gift_encrypt_kernel_table[grid[0], self.block_size](
                d_in, d_out, self.rkeys_device, np.int32(nblocks)
            )

        end_event.record()
        end_event.synchronize()
        kernel_seconds = cuda.event_elapsed_time(start_event, end_event) / 1000.0

        d2h_t0 = time.perf_counter()
        out = d_out.copy_to_host().astype(">u8").tobytes()
        d2h_seconds = time.perf_counter() - d2h_t0

        total_seconds = time.perf_counter() - t0
        transfer_seconds = h2d_seconds + d2h_seconds
        return out, GpuTiming(total_seconds, kernel_seconds, transfer_seconds)

    def encrypt_ctr(self, data: bytes, nonce: bytes | None = None) -> tuple[bytes, GpuTiming]:
        """Encrypt data in CTR mode using ECB(counter) keystream generation."""
        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        ctr_blocks = build_ctr_blocks(nblocks, 8, nonce=nonce)
        t0 = time.perf_counter()
        keystream, ecb_timing = self.encrypt_ecb(ctr_blocks)
        out = xor_bytes(data, keystream)
        total_seconds = time.perf_counter() - t0
        return out, GpuTiming(total_seconds, ecb_timing.kernel_seconds, ecb_timing.h2d_d2h_seconds)


def has_cuda_gpu() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False
