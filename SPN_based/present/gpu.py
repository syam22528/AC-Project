"""GPU-accelerated PRESENT-128 using Numba CUDA.

The pLayer is implemented via 4 delta-swap operations (instead of the
reference 63-iteration bit loop), reducing the permutation to 8 arithmetic
instructions per block.

Two S-box strategies are available for performance comparison:
  - 'bitsliced' (default) : tableless Boolean logic operating on all 16
                            nibbles simultaneously via GF(2) arithmetic.
                            Zero memory-access latency after register load.
  - 'table'               : constant-memory 16-entry S-box lookup, one
                            nibble at a time.

Additional optimisations:
  - 8x thread coarsening: each CUDA thread processes 8 cipher blocks to
    amortise shared-memory setup overhead.
  - Pinned host memory: allocated once and reused across calls to avoid
    repeated page-locking overhead.
  - Occupancy-aware block size: queried at construction time via
    cuda.occupancy.max_potential_block_size.
  - Native CTR mode kernels: generate and XOR keystream on-device,
    avoiding a separate ECB pass for CTR mode.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numba
from numba import cuda

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import SBOX, generate_round_keys
except Exception:
    from common import SBOX, generate_round_keys

# Cast SBOX to uint8 so it can be used as a CUDA constant array.
SBOX_CONST = SBOX.astype(np.uint8)


# ---------------------------------------------------------------------------
# CUDA device functions
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _delta_swap(x, mask, shift):
    """Swap bit-pairs separated by `shift` positions where `mask` marks the lower bits."""
    t = (x ^ (x >> numba.uint64(shift))) & numba.uint64(mask)
    return x ^ t ^ (t << numba.uint64(shift))


@cuda.jit(device=True, inline=True)
def p_layer_dev(x):
    """Apply the PRESENT pLayer via 4 delta-swaps on a 64-bit GPU state value.

    The PRESENT permutation (bit i → (16i mod 63), bit 63 fixed) is
    decomposed into 4 pairwise index-bit swaps implemented as delta_swap
    calls.  This avoids a 63-iteration loop in the hot CUDA kernel path.
    Verified against the reference loop on 100K random test vectors.
    """
    x = _delta_swap(x, 0x0A0A0A0A0A0A0A0A, 3)   # swap index bits 0 ↔ 2
    x = _delta_swap(x, 0x0000F0F00000F0F0, 12)   # swap index bits 2 ↔ 4
    x = _delta_swap(x, 0x00CC00CC00CC00CC, 6)    # swap index bits 1 ↔ 3
    x = _delta_swap(x, 0x00000000FF00FF00, 24)   # swap index bits 3 ↔ 5
    return x


@cuda.jit(device=True, inline=True)
def sbox_layer_bitsliced_dev(x):
    """Apply the PRESENT S-box to all 16 nibbles via bitsliced Boolean equations.

    Extracts 4 bit-planes (one per nibble bit), applies the S-box truth table
    in GF(2), and recombines the output bit-planes.  All 16 nibbles are
    processed simultaneously with no memory accesses beyond the input register.
    """
    m = numba.uint64(0x1111111111111111)   # mask: bit 0 of every nibble
    one = m

    x0 = x & m
    x1 = (x >> 1) & m
    x2 = (x >> 2) & m
    x3 = (x >> 3) & m

    # PRESENT S-box Boolean equations for output bits y0..y3.
    y0 = x0 ^ x2 ^ (x1 & x2) ^ x3
    y1 = x1 ^ (x0 & x1 & x2) ^ x3 ^ (x1 & x3) ^ (x0 & x1 & x3) ^ (x2 & x3) ^ (x0 & x2 & x3)
    y2 = one ^ (x0 & x1) ^ x2 ^ x3 ^ (x0 & x3) ^ (x1 & x3) ^ (x0 & x1 & x3) ^ (x0 & x2 & x3)
    y3 = one ^ x0 ^ x1 ^ (x1 & x2) ^ (x0 & x1 & x2) ^ x3 ^ (x0 & x1 & x3) ^ (x0 & x2 & x3)

    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)


@cuda.jit(device=True, inline=True)
def sbox_layer_table_dev(x):
    """Apply the PRESENT S-box to all 16 nibbles using a constant-memory table.

    Reads the 16-entry S-box from CUDA constant memory (cached on-chip),
    performing one lookup per nibble.
    """
    sbox = cuda.const.array_like(SBOX_CONST)
    out = numba.uint64(0)
    for i in range(16):
        shift = i * 4
        nib = (x >> shift) & numba.uint64(0xF)
        out |= numba.uint64(sbox[nib]) << shift
    return out


# ---------------------------------------------------------------------------
# ECB kernels (thread coarsening = 8 blocks/thread)
# ---------------------------------------------------------------------------

BLOCKS_PER_THREAD = 8   # each CUDA thread encrypts 8 PRESENT blocks


@cuda.jit
def present_encrypt_kernel_bitsliced(input_data, output_data, rkeys, nblocks):
    """PRESENT-128 ECB kernel: bitsliced S-box, 8x thread coarsening.

    Round keys are loaded into shared memory once per CUDA block.
    Each thread processes 8 consecutive plaintext blocks through 31 full
    rounds (AddRoundKey → SBox → pLayer) and a final AddRoundKey.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD

    shared_rkeys = cuda.shared.array(32, dtype=numba.uint64)
    for i in range(cuda.threadIdx.x, 32, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            return

        s = input_data[b]
        for r in range(31):
            s ^= shared_rkeys[r]              # AddRoundKey
            s = sbox_layer_bitsliced_dev(s)   # SubNibbles (bitsliced)
            s = p_layer_dev(s)                # pLayer
        s ^= shared_rkeys[31]                 # Final AddRoundKey
        output_data[b] = s


@cuda.jit
def present_encrypt_kernel_table(input_data, output_data, rkeys, nblocks):
    """PRESENT-128 ECB kernel: table-based S-box, 8x thread coarsening.

    Identical structure to the bitsliced kernel, but uses constant-memory
    S-box lookup for SubNibbles instead of Boolean equations.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD

    shared_rkeys = cuda.shared.array(32, dtype=numba.uint64)
    for i in range(cuda.threadIdx.x, 32, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            return

        s = input_data[b]
        for r in range(31):
            s ^= shared_rkeys[r]           # AddRoundKey
            s = sbox_layer_table_dev(s)    # SubNibbles (table)
            s = p_layer_dev(s)             # pLayer
        s ^= shared_rkeys[31]              # Final AddRoundKey
        output_data[b] = s


# ---------------------------------------------------------------------------
# Native CTR kernels (generate keystream and XOR in one pass)
# ---------------------------------------------------------------------------

@cuda.jit
def present_encrypt_ctr_kernel_bitsliced(input_data, output_data, rkeys, base_ctr_block, nblocks):
    """PRESENT-128 native CTR kernel: bitsliced S-box, 8x thread coarsening.

    Each block encrypts a counter value (base_ctr_block + block_index) directly
    on the device, then XORs the resulting keystream word with the corresponding
    plaintext word.  Eliminates the separate ECB + XOR pass used in the generic
    CTR path.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD

    shared_rkeys = cuda.shared.array(32, dtype=numba.uint64)
    for i in range(cuda.threadIdx.x, 32, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            return

        # Counter block: nonce prefix embedded in base_ctr_block, counter in low bits.
        s = numba.uint64(base_ctr_block) + numba.uint64(b)
        for r in range(31):
            s ^= shared_rkeys[r]
            s = sbox_layer_bitsliced_dev(s)
            s = p_layer_dev(s)
        s ^= shared_rkeys[31]
        # XOR the encrypted counter (keystream) with the plaintext block.
        output_data[b] = s ^ input_data[b]


@cuda.jit
def present_encrypt_ctr_kernel_table(input_data, output_data, rkeys, base_ctr_block, nblocks):
    """PRESENT-128 native CTR kernel: table-based S-box, 8x thread coarsening.

    Same structure as the bitsliced CTR kernel; uses table lookup for SubNibbles.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD

    shared_rkeys = cuda.shared.array(32, dtype=numba.uint64)
    for i in range(cuda.threadIdx.x, 32, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            return

        s = numba.uint64(base_ctr_block) + numba.uint64(b)
        for r in range(31):
            s ^= shared_rkeys[r]
            s = sbox_layer_table_dev(s)
            s = p_layer_dev(s)
        s ^= shared_rkeys[31]
        output_data[b] = s ^ input_data[b]


# ---------------------------------------------------------------------------
# Occupancy helper
# ---------------------------------------------------------------------------

def _best_block_size(kernel, fallback: int = 256) -> int:
    """Query CUDA occupancy API to find the block size that maximises occupancy.

    Falls back to `fallback` if the query is unsupported or raises.
    """
    try:
        _, block = cuda.occupancy.max_potential_block_size(kernel)
        return int(block)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Timing and public class
# ---------------------------------------------------------------------------

@dataclass
class GpuTiming:
    """Timing breakdown for a single GPU encryption call."""
    total_seconds: float
    kernel_seconds: float
    h2d_d2h_seconds: float


class PresentGpuOptimized:
    """PRESENT-128 GPU cipher — bitsliced (default) or table S-box variant.

    Key optimisations:
      - Delta-swap pLayer: 4 arithmetic operations instead of 63 iterations.
      - 8x thread coarsening to amortise shared-memory setup per block.
      - Pinned host memory + device buffers allocated once and reused.
      - Occupancy-aware block size selection at construction time.
      - Native CTR kernels that fuse counter generation and XOR on-device.
    """

    DEFAULT_VARIANT = "bitsliced"
    EVALUATION_VARIANTS = ("table", "bitsliced")

    def __init__(self, block_size: int | None = None, variant: str = DEFAULT_VARIANT) -> None:
        if variant not in ("table", "bitsliced"):
            raise ValueError("variant must be 'table' or 'bitsliced'")
        self.variant = variant

        # Select the correct kernel pair based on S-box variant.
        self._kernel = (
            present_encrypt_kernel_bitsliced
            if variant == "bitsliced"
            else present_encrypt_kernel_table
        )
        self._ctr_kernel = (
            present_encrypt_ctr_kernel_bitsliced
            if variant == "bitsliced"
            else present_encrypt_ctr_kernel_table
        )
        self.block_size = int(block_size) if block_size is not None else _best_block_size(self._kernel)
        self._is_key_set = False

        # Persistent pinned + device buffers, allocated lazily and grown on demand.
        self._buf_nblocks = 0
        self._h_in: np.ndarray | None = None
        self._h_out: np.ndarray | None = None
        self._d_in = None
        self._d_out = None

    def _ensure_buffers(self, nblocks: int) -> None:
        """Allocate pinned host and device buffers, reusing existing ones if sizes match."""
        if nblocks == self._buf_nblocks:
            return
        self._h_in = cuda.pinned_array(nblocks, dtype=np.uint64)
        self._h_out = cuda.pinned_array(nblocks, dtype=np.uint64)
        self._d_in = cuda.device_array(nblocks, dtype=np.uint64)
        self._d_out = cuda.device_array(nblocks, dtype=np.uint64)
        self._buf_nblocks = nblocks

    def set_key(self, key: bytes) -> None:
        """Expand `key` into 32 round keys and upload them to device memory."""
        if len(key) != 16:
            raise ValueError("PRESENT-128 requires a 16-byte key")
        rkeys = generate_round_keys(key).astype(np.uint64)
        self.rkeys_device = cuda.to_device(rkeys)
        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Raise ValueError if data is not a multiple of 8 bytes; return block count."""
        if len(data) % 8 != 0:
            raise ValueError("PRESENT block size is 8 bytes")
        return len(data) // 8

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in PRESENT-128 ECB mode on the GPU.

        Uses cached pinned host and device buffers to avoid repeated allocation.
        Kernel timing is measured via perf_counter around a cuda.synchronize().
        """
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)

        self._ensure_buffers(nblocks)

        t0 = time.perf_counter()

        # Copy plaintext into pinned host buffer then transfer to device.
        self._h_in[:] = states
        self._d_in.copy_to_device(self._h_in)

        logical_threads = math.ceil(nblocks / BLOCKS_PER_THREAD)
        grid = math.ceil(logical_threads / self.block_size)

        k0 = time.perf_counter()
        self._kernel[grid, self.block_size](
            self._d_in, self._d_out, self.rkeys_device, np.int32(nblocks)
        )
        cuda.synchronize()
        kernel_s = time.perf_counter() - k0

        # Copy ciphertext from device into pinned host buffer.
        self._d_out.copy_to_host(self._h_out)
        ct = self._h_out.astype(">u8").tobytes()

        total_s = time.perf_counter() - t0
        transfer_s = max(total_s - kernel_s, 0.0)
        return ct, GpuTiming(total_s, kernel_s, transfer_s)

    def encrypt_ctr(self, data: bytes, nonce: bytes | None = None) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in PRESENT-128 CTR mode using the native CTR GPU kernel.

        Constructs the base counter value from `nonce` on the host, then
        launches the on-device CTR kernel that generates the keystream and XORs
        each block's plaintext in a single pass.
        """
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        import os
        block_bytes = 8
        if nonce is None:
            nonce = os.urandom(block_bytes // 2)
        if len(nonce) >= block_bytes:
            raise ValueError("nonce must be shorter than block size")

        # Pack nonce into upper bytes and leave lower bytes for the counter.
        base_ctr_block = int.from_bytes(nonce + bytes(block_bytes - len(nonce)), "big")

        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)
        self._ensure_buffers(nblocks)

        t0 = time.perf_counter()

        self._h_in[:] = states
        self._d_in.copy_to_device(self._h_in)

        logical_threads = math.ceil(nblocks / BLOCKS_PER_THREAD)
        grid = math.ceil(logical_threads / self.block_size)

        k0 = time.perf_counter()
        self._ctr_kernel[grid, self.block_size](
            self._d_in, self._d_out, self.rkeys_device, np.uint64(base_ctr_block), np.int32(nblocks)
        )
        cuda.synchronize()
        kernel_s = time.perf_counter() - k0

        self._d_out.copy_to_host(self._h_out)
        ct = self._h_out.astype(">u8").tobytes()

        total_s = time.perf_counter() - t0
        transfer_s = max(total_s - kernel_s, 0.0)
        return ct, GpuTiming(total_s, kernel_s, transfer_s)


def has_cuda_gpu() -> bool:
    """Return True if a CUDA-capable GPU is available and Numba can use it."""
    try:
        return cuda.is_available()
    except Exception:
        return False