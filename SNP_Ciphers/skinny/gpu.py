from __future__ import annotations

"""GPU implementation of SKINNY-64-128 using Numba CUDA.

Provides table and bitsliced S-box variants for performance comparison.

Optimizations:
- 8-bit table S-box expands 4-bit SBOX to dual-nibble lookups (16 iters -> 8)
- Bitsliced variant utilizes pure ALU without cache boundaries.
- Thread Coarsening set to 4 blocks per thread.
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
    """Precompute 8-bit table that evaluates 2 S-boxes at once."""
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lo = SBOX[i & 0xF]
        hi = SBOX[i >> 4]
        table[i] = lo | (hi << 4)
    return table

SBOX8_CONST = _compute_sbox8()

# Thread coarsening factor: chunks of blocks handled per thread
BLOCKS_PER_THREAD = 4

@cuda.jit(device=True)
def sbox_layer_table_dev_8bit(x):
    """Apply dual-nibble table S-box via constant memory."""
    sbox8 = cuda.const.array_like(SBOX8_CONST)
    out = numba.uint64(0)
    for i in range(8):
        shift = i * 8
        byte_val = (x >> shift) & numba.uint64(0xFF)
        out |= numba.uint64(sbox8[byte_val]) << shift
    return out


@cuda.jit(device=True)
def sbox_layer_bitsliced_dev(x):
    """Apply SKINNY S-box via bitsliced boolean equations."""
    x = ~x
    x ^= (((x >> 3) & (x >> 2)) & numba.uint64(0x1111111111111111))
    x ^= (((x << 1) & (x << 2)) & numba.uint64(0x8888888888888888))
    x ^= (((x << 1) & (x << 2)) & numba.uint64(0x4444444444444444))
    x ^= (((x >> 2) & (x << 1)) & numba.uint64(0x2222222222222222))
    x = ~x
    return ((x >> 1) & numba.uint64(0x7777777777777777)) | ((x << 3) & numba.uint64(0x8888888888888888))


@cuda.jit(device=True)
def ror16_dev(x, c):
    """Rotate a 16-bit value right by c bits on device."""
    return numba.uint16(((x >> c) | (x << (16 - c))) & 0xFFFF)


@cuda.jit(device=True)
def round_linear_dev(s):
    """Apply SKINNY linear layer (ShiftRows + MixColumns)."""
    row0 = numba.uint16(s & numba.uint64(0xFFFF))
    row1 = numba.uint16((s >> 16) & numba.uint64(0xFFFF))
    row2 = numba.uint16((s >> 32) & numba.uint64(0xFFFF))
    row3 = numba.uint16((s >> 48) & numba.uint64(0xFFFF))

    row1 = ror16_dev(row1, 4)
    row2 = ror16_dev(row2, 8)
    row3 = ror16_dev(row3, 12)

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

@cuda.jit
def skinny_encrypt_kernel_table(input_data, output_data, rkeys, nblocks):
    """GPU kernel for table-based SKINNY encryption."""
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
            s = sbox_layer_table_dev_8bit(s)
            lo = numba.uint32(s & numba.uint64(0xFFFFFFFF))
            lo ^= shared_rkeys[r]
            s = (s & numba.uint64(0xFFFFFFFF00000000)) | numba.uint64(lo)
            s ^= numba.uint64(0x0000002000000000)
            s = round_linear_dev(s)
        output_data[b] = s


@cuda.jit
def skinny_encrypt_kernel_bitsliced(input_data, output_data, rkeys, nblocks):
    """GPU kernel for bitsliced SKINNY encryption."""
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
            s = sbox_layer_bitsliced_dev(s)
            lo = numba.uint32(s & numba.uint64(0xFFFFFFFF))
            lo ^= shared_rkeys[r]
            s = (s & numba.uint64(0xFFFFFFFF00000000)) | numba.uint64(lo)
            s ^= numba.uint64(0x0000002000000000)
            s = round_linear_dev(s)
        output_data[b] = s

@dataclass
class GpuTiming:
    """Timing breakdown for one GPU ECB encryption call."""
    total_seconds: float
    kernel_seconds: float
    h2d_d2h_seconds: float

class SkinnyGpuOptimized:
    """SKINNY-64-128 GPU with default and evaluation variants."""

    DEFAULT_VARIANT = "table"
    EVALUATION_VARIANTS = ("table", "bitsliced")

    def __init__(self, block_size: int = 256, variant: str = DEFAULT_VARIANT) -> None:
        self.block_size = int(block_size)
        if variant not in ("table", "bitsliced"):
            raise ValueError("variant must be 'table' or 'bitsliced'")
        self.variant = variant
        self._is_key_set = False

    def set_key(self, key: bytes) -> None:
        """Set 128-bit key and upload round subkeys to GPU memory."""
        rkeys = generate_round_subkeys(key).astype(np.uint32)
        self.rkeys_device = cuda.to_device(rkeys)
        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Validate input length and return number of 8-byte blocks."""
        if len(data) % 8 != 0:
            raise ValueError("SKINNY block size is 8 bytes")
        return len(data) // 8

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt ECB data on GPU and return ciphertext + timing metrics."""
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        # SKINNY blocks use native endianness internally corresponding with _load_u32_le on creation
        states = np.frombuffer(data, dtype=np.uint64).astype(np.uint64)

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
