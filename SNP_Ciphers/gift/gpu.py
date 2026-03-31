from __future__ import annotations

"""GPU implementation of GIFT-64-128 using Numba CUDA.

Provides two S-box strategies for performance comparison:
  - 'table'     : fused SubNibbles + PermBits using a 256-entry constant-memory
                  scatter table.  Replaces 80 loop iterations (16 S-box + 64
                  P-box) with 16 table lookups.
  - 'bitsliced' : tableless Boolean equations for SubNibbles, followed by
                  a nibble scatter table for PermBits.  Better on ALU-saturated
                  configurations where constant-cache bandwidth is a bottleneck.

Both kernels use 4 blocks per thread (thread coarsening) and load round keys
into shared memory to reduce global memory traffic.
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
# Host-side scatter table construction
# ---------------------------------------------------------------------------

def _compute_pbox_scatter() -> np.ndarray:
    """Build a 256-entry constant-memory table for the GIFT permutation layer.

    PBOX_SCATTER[k*16 + v] is the 64-bit word that results from scattering
    the 4 bits of nibble value `v` (coming from input nibble position `k`)
    to their correct output bit positions according to PBOX.

    Used by the bitsliced kernel to replace a 64-iteration bit loop with
    16 table lookups.
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
    """Build a 256-entry constant-memory table that fuses SubNibbles + PermBits.

    SP_SCATTER[k*16 + v] = pbox_scatter(k, SBOX[v]).

    Using this table, a single 16-iteration loop simultaneously performs
    SubNibbles and PermBits, replacing separate 16-iter + 64-iter loops.
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

# Number of GIFT blocks processed by each CUDA thread (thread coarsening).
BLOCKS_PER_THREAD = 4


# ---------------------------------------------------------------------------
# CUDA device functions
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def sp_layer_table_dev(x):
    """Apply fused SubNibbles + PermBits via the SP_SCATTER constant-memory table.

    Iterates over 16 nibble positions; for each position k, extracts the
    nibble, looks up SP_SCATTER[k*16 + nibble], and ORs the result into the
    output word.  The table already encodes both substitution and permutation.
    """
    sp = cuda.const.array_like(SP_SCATTER_CONST)
    out = numba.uint64(0)
    for k in range(16):
        nib = (x >> numba.uint64(k * 4)) & numba.uint64(0xF)
        out |= sp[k * 16 + nib]
    return out


@cuda.jit(device=True)
def sbox_layer_bitsliced_dev(x):
    """Apply the GIFT S-box to all 16 nibbles simultaneously via Boolean equations.

    Extracts the 4 bit-planes of the 64-bit state (bits 0, 1, 2, 3 of every
    nibble), applies the S-box logic in GF(2) arithmetic, and recombines the
    output bit-planes.
    """
    m = numba.uint64(0x1111111111111111)  # isolates every 4th bit (bit 0 of each nibble)
    one = m

    x0 = x & m
    x1 = (x >> 1) & m
    x2 = (x >> 2) & m
    x3 = (x >> 3) & m

    # Boolean equations for the GIFT S-box output bits (y0..y3).
    y0 = one ^ x0 ^ x1 ^ (x0 & x1) ^ x2 ^ x3
    y1 = x0 ^ (x0 & x1) ^ x2 ^ (x0 & x2) ^ x3
    y2 = x1 ^ x2 ^ (x0 & x3) ^ (x1 & x3) ^ (x1 & x2 & x3)
    y3 = x0 ^ (x1 & x3) ^ (x0 & x2 & x3)

    return y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)


@cuda.jit(device=True)
def p_layer_scatter_dev(x):
    """Apply GIFT PermBits via the PBOX_SCATTER constant-memory table.

    Replaces the 64-iteration bit loop with 16 nibble-level lookups,
    each returning the correctly positioned output bits for that nibble.
    """
    pt = cuda.const.array_like(PBOX_SCATTER_CONST)
    out = numba.uint64(0)
    for k in range(16):
        nib = (x >> numba.uint64(k * 4)) & numba.uint64(0xF)
        out |= pt[k * 16 + nib]
    return out


# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------

@cuda.jit
def gift_encrypt_kernel_table(input_data, output_data, rkeys, nblocks):
    """GIFT-64-128 ECB kernel using the fused SP scatter table.

    Round keys are loaded into shared memory so all threads in a block
    share a single copy.  Each thread processes BLOCKS_PER_THREAD blocks.
    Each round: SP_SCATTER table lookup → XOR round key.
    """
    tid = cuda.grid(1)
    first = tid * BLOCKS_PER_THREAD
    shared_rkeys = cuda.shared.array(28, dtype=numba.uint64)

    # Cooperatively load all 28 round keys into shared memory.
    for i in range(cuda.threadIdx.x, 28, cuda.blockDim.x):
        shared_rkeys[i] = rkeys[i]
    cuda.syncthreads()

    for it in range(BLOCKS_PER_THREAD):
        b = first + it
        if b >= nblocks:
            continue

        s = input_data[b]
        for r in range(28):
            s = sp_layer_table_dev(s)      # Fused SubNibbles + PermBits
            s ^= shared_rkeys[r]           # AddRoundKey + round constant (combined mask)
        output_data[b] = s


@cuda.jit
def gift_encrypt_kernel_bitsliced(input_data, output_data, rkeys, nblocks):
    """GIFT-64-128 ECB kernel using bitsliced S-box and scatter permutation.

    Round keys are loaded into shared memory.  Each round applies:
    bitsliced SubNibbles → scatter PermBits → XOR round key.
    """
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
            s = sbox_layer_bitsliced_dev(s)  # SubNibbles via Boolean equations
            s = p_layer_scatter_dev(s)       # PermBits via nibble scatter table
            s ^= shared_rkeys[r]             # AddRoundKey + round constant
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


class GiftGpuOptimized:
    """GIFT-64-128 GPU cipher supporting table and bitsliced S-box variants.

    Use 'table' variant for best throughput on most hardware.
    Use 'bitsliced' variant as an alternative for ALU-heavy comparisons.
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
        """Compute 28 round masks from `key` and upload them to GPU device memory."""
        round_masks = generate_round_masks(key).astype(np.uint64)
        self.rkeys_device = cuda.to_device(round_masks)
        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Raise ValueError if data is not a multiple of 8; return block count."""
        if len(data) % 8 != 0:
            raise ValueError("GIFT block size is 8 bytes")
        return len(data) // 8

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in GIFT-64-128 ECB mode on the GPU.

        Transfers plaintext host→device, launches the selected CUDA kernel
        with CUDA-event timing, copies ciphertext device→host.
        Returns (ciphertext, GpuTiming).
        """
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        # Interpret 8-byte blocks as big-endian 64-bit integers.
        states = np.frombuffer(data, dtype=">u8").astype(np.uint64)

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
        """Encrypt `data` in GIFT-64-128 CTR mode on the GPU.

        Counter blocks are built on the host and encrypted via ECB on the GPU
        to form a keystream, which is XORed with the plaintext.
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
