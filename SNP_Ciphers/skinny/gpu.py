from __future__ import annotations

"""GPU implementation of SKINNY-64-128 using Numba CUDA.

Two S-box strategies are provided for performance comparison:
  - 'table'     : dual-nibble 8-bit constant-memory table that evaluates two
                  S-box lookups per byte in a single table access, halving the
                  number of memory operations compared to a 4-bit nibble table.
  - 'bitsliced' : tableless Boolean equations operating on all 16 nibbles
                  simultaneously with zero memory-access latency.

Both kernels apply 4x thread coarsening (4 blocks per thread) and cache
all 36 round subkeys in shared memory to reduce global-memory traffic.
"""

import time
from dataclasses import dataclass

import numpy as np
import numba
from numba import cuda

from ctr_utils import build_ctr_blocks, xor_bytes

try:
    from .common import SBOX, generate_round_subkeys
except Exception:
    from common import SBOX, generate_round_subkeys


def _compute_sbox8() -> np.ndarray:
    """Precompute a 256-entry table that evaluates two 4-bit S-boxes per byte.

    SBOX8[byte] = SBOX[byte & 0xF] | (SBOX[byte >> 4] << 4).
    This lets the GPU process pairs of nibbles with a single constant-memory
    lookup, reducing the 16-iteration per-nibble loop to 8 iterations.
    """
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lo = SBOX[i & 0xF]
        hi = SBOX[i >> 4]
        table[i] = lo | (hi << 4)
    return table


SBOX8_CONST = _compute_sbox8()

# Number of SKINNY-64 blocks each CUDA thread processes (thread coarsening).
BLOCKS_PER_THREAD = 4


# ---------------------------------------------------------------------------
# CUDA device functions
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def sbox_layer_table_dev_8bit(x):
    """Apply the SKINNY S-box to all 16 nibbles using the dual-nibble 8-bit table.

    Processes the 64-bit state byte by byte (8 iterations), applying SBOX8
    from constant memory to substitute both nibbles of each byte at once.
    """
    sbox8 = cuda.const.array_like(SBOX8_CONST)
    out = numba.uint64(0)
    for i in range(8):
        shift = i * 8
        byte_val = (x >> shift) & numba.uint64(0xFF)
        out |= numba.uint64(sbox8[byte_val]) << shift
    return out


@cuda.jit(device=True)
def sbox_layer_bitsliced_dev(x):
    """Apply the SKINNY S-box to all 16 nibbles via bitsliced Boolean equations.

    Inverts the input, applies four rounds of nibble-level Boolean operations
    that implement the SKINNY S-box truth table in GF(2), then inverts and
    rotates the output nibbles right by 1 bit.
    """
    x = ~x
    x ^= (((x >> 3) & (x >> 2)) & numba.uint64(0x1111111111111111))
    x ^= (((x << 1) & (x << 2)) & numba.uint64(0x8888888888888888))
    x ^= (((x << 1) & (x << 2)) & numba.uint64(0x4444444444444444))
    x ^= (((x >> 2) & (x << 1)) & numba.uint64(0x2222222222222222))
    x = ~x
    # Rotate output nibbles right by 1 position within each 4-bit cell.
    return ((x >> 1) & numba.uint64(0x7777777777777777)) | ((x << 3) & numba.uint64(0x8888888888888888))


@cuda.jit(device=True)
def ror16_dev(x, c):
    """Rotate a 16-bit value right by `c` bits (device helper for ShiftRows)."""
    return numba.uint16(((x >> c) | (x << (16 - c))) & 0xFFFF)


@cuda.jit(device=True)
def round_linear_dev(s):
    """Apply the SKINNY-64 linear layer: ShiftRows then MixColumns.

    The 64-bit state is split into four 16-bit rows:
      row0 = bits [15:0] (no rotation)
      row1 = bits [31:16] (rotated right by 4 nibbles)
      row2 = bits [47:32] (rotated right by 8 nibbles)
      row3 = bits [63:48] (rotated right by 12 nibbles)
    MixColumns then combines them with the SKINNY mixing matrix.
    """
    row0 = numba.uint16(s & numba.uint64(0xFFFF))
    row1 = numba.uint16((s >> 16) & numba.uint64(0xFFFF))
    row2 = numba.uint16((s >> 32) & numba.uint64(0xFFFF))
    row3 = numba.uint16((s >> 48) & numba.uint64(0xFFFF))

    # ShiftRows: rotate each row right by its index × 4 bit positions.
    row1 = ror16_dev(row1, 4)
    row2 = ror16_dev(row2, 8)
    row3 = ror16_dev(row3, 12)

    # MixColumns: mix using the SKINNY mixing matrix.
    row1x = numba.uint16(row1 ^ row2)
    row2x = numba.uint16(row2 ^ row0)
    temp = numba.uint16(row3 ^ row2x)

    row3 = row2x
    row2 = row1x
    row1 = row0
    row0 = temp

    return (
        numba.uint64(row0)
        | (numba.uint64(row1) << 16)
        | (numba.uint64(row2) << 32)
        | (numba.uint64(row3) << 48)
    )


# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------

@cuda.jit
def skinny_encrypt_kernel_table(input_data, output_data, rkeys, nblocks):
    """SKINNY-64-128 ECB kernel: dual-nibble table S-box, 4x thread coarsening.

    Round subkeys are loaded into shared memory once per CUDA block.
    Per round: table SubCells → AddConstants → AddRoundKey → linear layer.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD
    shared_rkeys = cuda.shared.array(36, dtype=numba.uint32)

    # Cooperatively load all 36 round subkeys into shared memory.
    for i in range(cuda.threadIdx.x, 36, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            continue

        s = input_data[b]
        for r in range(36):
            s = sbox_layer_table_dev_8bit(s)                       # SubCells
            lo = numba.uint32(s & numba.uint64(0xFFFFFFFF))
            lo ^= shared_rkeys[r]                                   # AddRoundKey (lower 32 bits)
            s = (s & numba.uint64(0xFFFFFFFF00000000)) | numba.uint64(lo)
            s ^= numba.uint64(0x0000002000000000)                   # AddConstants
            s = round_linear_dev(s)                                 # ShiftRows + MixColumns
        output_data[b] = s


@cuda.jit
def skinny_encrypt_kernel_bitsliced(input_data, output_data, rkeys, nblocks):
    """SKINNY-64-128 ECB kernel: bitsliced S-box, 4x thread coarsening.

    Identical round structure to the table kernel, using bitsliced SubCells
    instead of constant-memory lookups.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD
    shared_rkeys = cuda.shared.array(36, dtype=numba.uint32)

    for i in range(cuda.threadIdx.x, 36, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            continue

        s = input_data[b]
        for r in range(36):
            s = sbox_layer_bitsliced_dev(s)                        # SubCells (bitsliced)
            lo = numba.uint32(s & numba.uint64(0xFFFFFFFF))
            lo ^= shared_rkeys[r]                                   # AddRoundKey
            s = (s & numba.uint64(0xFFFFFFFF00000000)) | numba.uint64(lo)
            s ^= numba.uint64(0x0000002000000000)                   # AddConstants
            s = round_linear_dev(s)                                 # ShiftRows + MixColumns
        output_data[b] = s


# ---------------------------------------------------------------------------
# Host wrapper class
# ---------------------------------------------------------------------------

@dataclass
class GpuTiming:
    """Timing breakdown for a single GPU ECB or CTR encryption call."""
    total_seconds: float
    kernel_seconds: float
    h2d_d2h_seconds: float


class SkinnyGpuOptimized:
    """SKINNY-64-128 GPU cipher supporting table and bitsliced S-box variants.

    Use 'table' for best throughput on CUDA hardware with fast constant caches.
    Use 'bitsliced' as an ALU-only alternative for comparison benchmarks.
    """

    DEFAULT_VARIANT = "table"
    EVALUATION_VARIANTS = ("table", "bitsliced")

    def __init__(self, block_size: int = 256, variant: str = DEFAULT_VARIANT) -> None:
        self.block_size = int(block_size)
        if variant not in ("table", "bitsliced"):
            raise ValueError("variant must be 'table' or 'bitsliced'")
        self.variant = variant
        self._is_key_set = False

    def set_key(self, key: bytes) -> None:
        """Compute 36 round subkeys from `key` and upload them to GPU device memory."""
        rkeys = generate_round_subkeys(key).astype(np.uint32)
        self.rkeys_device = cuda.to_device(rkeys)
        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Raise ValueError if data is not a multiple of 8 bytes; return block count."""
        if len(data) % 8 != 0:
            raise ValueError("SKINNY block size is 8 bytes")
        return len(data) // 8

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in SKINNY-64-128 ECB mode on the GPU.

        Transfers plaintext host→device, launches the CUDA kernel with
        CUDA-event timing, then copies ciphertext device→host.
        Returns (ciphertext, GpuTiming).
        """
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        # Load state as native-endian uint64 (consistent with _load_u32_le in the key schedule).
        states = np.frombuffer(data, dtype=np.uint64).astype(np.uint64)

        t0 = time.perf_counter()

        h2d_t0 = time.perf_counter()
        d_in = cuda.to_device(states)
        d_out = cuda.device_array_like(d_in)
        h2d_seconds = time.perf_counter() - h2d_t0

        # Round up thread count to cover all blocks with BLOCKS_PER_THREAD coarsening.
        logical_threads = (nblocks + BLOCKS_PER_THREAD - 1) // BLOCKS_PER_THREAD
        grid = ((logical_threads + self.block_size - 1) // self.block_size,)

        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record()

        if self.variant == "bitsliced":
            skinny_encrypt_kernel_bitsliced[grid[0], self.block_size](
                d_in, d_out, self.rkeys_device, np.int32(nblocks)
            )
        else:
            skinny_encrypt_kernel_table[grid[0], self.block_size](
                d_in, d_out, self.rkeys_device, np.int32(nblocks)
            )

        end_event.record()
        end_event.synchronize()
        kernel_seconds = cuda.event_elapsed_time(start_event, end_event) / 1000.0

        d2h_t0 = time.perf_counter()
        out = d_out.copy_to_host().astype(np.uint64).tobytes()
        d2h_seconds = time.perf_counter() - d2h_t0

        total_seconds = time.perf_counter() - t0
        transfer_seconds = h2d_seconds + d2h_seconds
        return out, GpuTiming(total_seconds, kernel_seconds, transfer_seconds)

    def encrypt_ctr(self, data: bytes, nonce: bytes | None = None) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in SKINNY-64-128 CTR mode on the GPU.

        Counter blocks are built on the host and encrypted via ECB on the GPU
        to produce a keystream, which is XORed with the plaintext.
        """
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
    """Return True if a CUDA-capable GPU is available and Numba can use it."""
    try:
        return cuda.is_available()
    except Exception:
        return False
