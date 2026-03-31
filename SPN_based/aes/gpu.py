"""GPU-accelerated AES-128 using Numba CUDA.

The default kernel stores the S-box in shared memory (256 bytes per block)
and loads round keys from device global memory.  An optional constant-memory
round-key variant is also available; it trades key flexibility for lower
memory-access latency on hardware with fast constant caches.

Thread coarsening is set to 2 AES blocks per thread to improve arithmetic
intensity and hide global-memory latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numba
from numba import cuda

from ctr_utils import build_ctr_blocks, xor_bytes


# AES-128 S-box used for host-side key expansion and uploaded to device.
SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)

# AES key-schedule round constants (rounds 1-10, one byte each).
RCON = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint8)


def expand_key_128(key: bytes) -> np.ndarray:
    """Expand a 16-byte AES-128 key into a flat 176-byte schedule (11 round keys).

    The schedule is stored as a 1-D uint8 array; round key r occupies bytes
    [r*16 : r*16+16].  Implements the standard AES key expansion algorithm
    (FIPS 197 Section 5.2).
    """
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")

    expanded = np.zeros(176, dtype=np.uint8)
    expanded[:16] = np.frombuffer(key, dtype=np.uint8)

    bytes_generated = 16
    rcon_iter = 0

    while bytes_generated < 176:
        t0 = expanded[bytes_generated - 4]
        t1 = expanded[bytes_generated - 3]
        t2 = expanded[bytes_generated - 2]
        t3 = expanded[bytes_generated - 1]

        if bytes_generated % 16 == 0:
            # RotWord: rotate the 4-byte word left by one byte.
            t0, t1, t2, t3 = t1, t2, t3, t0
            # SubWord: apply S-box to each byte.
            t0, t1, t2, t3 = int(SBOX[t0]), int(SBOX[t1]), int(SBOX[t2]), int(SBOX[t3])
            # XOR the first byte with the round constant.
            t0 ^= int(RCON[rcon_iter])
            rcon_iter += 1

        expanded[bytes_generated + 0] = expanded[bytes_generated - 16] ^ t0
        expanded[bytes_generated + 1] = expanded[bytes_generated - 15] ^ t1
        expanded[bytes_generated + 2] = expanded[bytes_generated - 14] ^ t2
        expanded[bytes_generated + 3] = expanded[bytes_generated - 13] ^ t3
        bytes_generated += 4

    return expanded


# ---------------------------------------------------------------------------
# CUDA device functions (run inside kernel threads)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def xtime_dev(x):
    """Multiply a GF(2^8) byte by 2 using the AES reduction polynomial (0x11B).

    Implemented as branchless: shift left by 1, then XOR 0x1B if the
    original MSB was set, then mask to 8 bits.
    """
    return numba.uint8(((x << 1) ^ (((x >> 7) & 1) * 0x1B)) & 0xFF)


@cuda.jit(device=True)
def add_round_key_offset_dev(s, rkeys, offset):
    """XOR 16 bytes of state `s` with the round key starting at `rkeys[offset]`."""
    for i in range(16):
        s[i] ^= rkeys[offset + i]


@cuda.jit(device=True)
def sub_bytes_table_dev(s, table):
    """Apply SubBytes to all 16 state bytes using the provided S-box lookup table."""
    for i in range(16):
        s[i] = table[s[i]]


@cuda.jit(device=True)
def shift_rows_dev(s):
    """Apply AES ShiftRows to state `s` stored in column-major byte order.

    The 4×4 AES state is laid out column-major in `s[16]`:
      s[0]  s[4]  s[8]  s[12]   (row 0 — no shift)
      s[1]  s[5]  s[9]  s[13]   (row 1 — cyclic left shift by 1)
      s[2]  s[6]  s[10] s[14]   (row 2 — cyclic left shift by 2)
      s[3]  s[7]  s[11] s[15]   (row 3 — cyclic left shift by 3)
    """
    # Row 1: rotate left by 1 byte.
    t = s[1]
    s[1] = s[5]
    s[5] = s[9]
    s[9] = s[13]
    s[13] = t

    # Row 2: rotate left by 2 bytes (swap pairs).
    t = s[2]
    s[2] = s[10]
    s[10] = t
    t = s[6]
    s[6] = s[14]
    s[14] = t

    # Row 3: rotate left by 3 bytes (equivalent to rotate right by 1).
    t = s[3]
    s[3] = s[15]
    s[15] = s[11]
    s[11] = s[7]
    s[7] = t


@cuda.jit(device=True)
def mix_columns_xtime_dev(s):
    """Apply AES MixColumns to each of the 4 columns using branchless xtime.

    Each column [a0, a1, a2, a3] is multiplied by the MixColumns matrix
    in GF(2^8):
      [2 3 1 1]
      [1 2 3 1]
      [1 1 2 3]
      [3 1 1 2]
    The standard compound trick reduces this to three xtime() calls per column.
    """
    for c in range(4):
        i = c * 4
        a0 = s[i + 0]
        a1 = s[i + 1]
        a2 = s[i + 2]
        a3 = s[i + 3]
        t = numba.uint8(a0 ^ a1 ^ a2 ^ a3)
        u = a0
        s[i + 0] ^= t ^ xtime_dev(numba.uint8(a0 ^ a1))
        s[i + 1] ^= t ^ xtime_dev(numba.uint8(a1 ^ a2))
        s[i + 2] ^= t ^ xtime_dev(numba.uint8(a2 ^ a3))
        s[i + 3] ^= t ^ xtime_dev(numba.uint8(a3 ^ u))


# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------

@cuda.jit
def aes_optimized_kernel(input_data, output_data, sbox, rkeys, nblocks):
    """AES-128 ECB encryption kernel: shared-memory S-box, global round keys.

    Each CUDA block cooperatively loads the 256-byte S-box and 176-byte key
    schedule into shared memory before processing blocks.  Each thread then
    handles 2 AES blocks (thread coarsening) to hide memory latency and
    improve arithmetic-to-bandwidth ratio.
    """
    # Cooperatively load S-box and round keys into on-chip shared memory.
    shared_sbox = cuda.shared.array(256, dtype=numba.uint8)
    shared_rkeys = cuda.shared.array(176, dtype=numba.uint8)
    for idx in range(cuda.threadIdx.x, 256, cuda.blockDim.x):
        shared_sbox[idx] = sbox[idx]
    for idx in range(cuda.threadIdx.x, 176, cuda.blockDim.x):
        shared_rkeys[idx] = rkeys[idx]
    cuda.syncthreads()

    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    first_block = tid * 2

    # Each thread encrypts 2 consecutive AES blocks.
    for it in range(2):
        b = first_block + it
        if b >= nblocks:
            continue

        base = b * 16
        s = cuda.local.array(16, dtype=numba.uint8)

        # Load one 16-byte plaintext block into local register state.
        for i in range(16):
            s[i] = input_data[base + i]

        # Initial AddRoundKey (round key 0).
        add_round_key_offset_dev(s, shared_rkeys, 0)

        # Rounds 1–9: SubBytes → ShiftRows → MixColumns → AddRoundKey.
        for round_num in range(1, 10):
            sub_bytes_table_dev(s, shared_sbox)
            shift_rows_dev(s)
            mix_columns_xtime_dev(s)
            add_round_key_offset_dev(s, shared_rkeys, round_num * 16)

        # Final round: SubBytes → ShiftRows → AddRoundKey (no MixColumns).
        sub_bytes_table_dev(s, shared_sbox)
        shift_rows_dev(s)
        add_round_key_offset_dev(s, shared_rkeys, 160)

        # Write ciphertext block back to global memory.
        for i in range(16):
            output_data[base + i] = s[i]


def make_aes_constkeys_shared_kernel(rkeys_const_np: np.ndarray):
    """Return a key-specialised CUDA kernel that stores round keys in constant memory.

    Captures `rkeys_const_np` at compile time via cuda.const.array_like so
    the GPU reads round keys from the constant cache instead of global memory.
    This lowers latency for warp-uniform accesses but requires one compiled
    kernel instance per distinct key.  Results are cached in AesGpuOptimized
    to avoid re-JIT-compilation on key reuse.
    """

    @cuda.jit
    def aes_constkeys_shared_kernel(input_data, output_data, sbox, nblocks):
        shared_sbox = cuda.shared.array(256, dtype=numba.uint8)
        # Round keys in constant memory — baked into this kernel at JIT time.
        const_rkeys = cuda.const.array_like(rkeys_const_np)

        for idx in range(cuda.threadIdx.x, 256, cuda.blockDim.x):
            shared_sbox[idx] = sbox[idx]
        cuda.syncthreads()

        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        first_block = tid * 2

        for it in range(2):
            b = first_block + it
            if b >= nblocks:
                continue

            base = b * 16
            s = cuda.local.array(16, dtype=numba.uint8)

            for i in range(16):
                s[i] = input_data[base + i]

            add_round_key_offset_dev(s, const_rkeys, 0)

            for round_num in range(1, 10):
                sub_bytes_table_dev(s, shared_sbox)
                shift_rows_dev(s)
                mix_columns_xtime_dev(s)
                add_round_key_offset_dev(s, const_rkeys, round_num * 16)

            sub_bytes_table_dev(s, shared_sbox)
            shift_rows_dev(s)
            add_round_key_offset_dev(s, const_rkeys, 160)

            for i in range(16):
                output_data[base + i] = s[i]

    return aes_constkeys_shared_kernel


# ---------------------------------------------------------------------------
# Host-side wrapper class
# ---------------------------------------------------------------------------

@dataclass
class GpuTiming:
    """Wall-clock and hardware timing breakdown for a single GPU encryption call."""
    total_seconds: float
    kernel_seconds: float
    h2d_d2h_seconds: float


class AesGpuOptimized:
    """AES-128 GPU cipher used by benchmarks.

    Supports two key-storage modes:
      - 'global'    : round keys passed as a kernel argument (more flexible).
      - 'constkeys' : round keys baked into constant memory (potentially faster
                      on hardware with a large constant cache).

    The S-box is always placed in shared memory for low-latency access.
    """

    DEFAULT_VARIANT = "shared"
    DEFAULT_KEY_MODE = "global"
    EVALUATION_KEY_MODES = ("global", "constkeys")

    # Class-level cache mapping (variant, key_bytes) -> compiled kernel.
    _constkeys_kernel_cache: dict[tuple[str, bytes], Any] = {}

    def __init__(self, block_size: int | None = None, variant: str = DEFAULT_VARIANT, key_mode: str = DEFAULT_KEY_MODE) -> None:
        self.block_size = block_size if block_size is not None else 128
        if variant != "shared":
            raise ValueError("variant must be 'shared'")
        if key_mode not in ("global", "constkeys"):
            raise ValueError("key_mode must be 'global' or 'constkeys'")
        self.variant = variant
        self.key_mode = key_mode

        if self.key_mode == "global":
            self.kernel = aes_optimized_kernel
        else:
            # For constkeys, the kernel is bound to a specific key in set_key().
            self.kernel = None

        self.sbox_device = cuda.to_device(SBOX)
        self._is_key_set = False

    def set_key(self, key: bytes) -> None:
        """Expand `key` and upload the round-key schedule to device memory.

        For constkeys mode, also compiles (or retrieves a cached) kernel that
        embeds this key in CUDA constant memory.
        """
        rkeys = expand_key_128(key)
        self.rkeys_device = cuda.to_device(rkeys)

        if self.key_mode == "constkeys":
            cache_key = (self.variant, bytes(key))
            kernel = self._constkeys_kernel_cache.get(cache_key)
            if kernel is None:
                kernel = make_aes_constkeys_shared_kernel(rkeys.copy())
                self._constkeys_kernel_cache[cache_key] = kernel
            self.kernel = kernel

        self._is_key_set = True

    @staticmethod
    def _validate_data(data: bytes) -> int:
        """Raise ValueError if data is not a multiple of 16 bytes; return block count."""
        if len(data) % 16 != 0:
            raise ValueError("Input length must be a multiple of 16 bytes")
        return len(data) // 16

    def encrypt_ecb(self, data: bytes) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in AES-128 ECB mode on the GPU.

        Transfers plaintext host→device, launches the CUDA kernel with CUDA-event
        timing for accurate kernel-only measurement, then copies ciphertext
        device→host.  Returns (ciphertext, GpuTiming).
        """
        if not self._is_key_set:
            raise RuntimeError("Call set_key(key) before encryption")

        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        arr_in_np = np.frombuffer(data, dtype=np.uint8).copy()
        t0 = time.perf_counter()

        # Host → Device transfer, timed separately.
        h2d_t0 = time.perf_counter()
        d_in = cuda.to_device(arr_in_np)
        d_out = cuda.device_array_like(d_in)
        h2d_seconds = time.perf_counter() - h2d_t0

        # Each thread handles 2 blocks; round up to cover all blocks.
        logical_threads = (nblocks + 1) // 2
        grid_size = (logical_threads + self.block_size - 1) // self.block_size

        # Use CUDA events for hardware-accurate kernel timing.
        start_event = cuda.event()
        end_event = cuda.event()
        start_event.record()

        if self.key_mode == "global":
            self.kernel[grid_size, self.block_size](
                d_in, d_out, self.sbox_device, self.rkeys_device, np.int32(nblocks)
            )
        else:
            self.kernel[grid_size, self.block_size](
                d_in, d_out, self.sbox_device, np.int32(nblocks)
            )

        end_event.record()
        end_event.synchronize()
        kernel_seconds = cuda.event_elapsed_time(start_event, end_event) / 1000.0

        # Device → Host transfer, timed separately.
        d2h_t0 = time.perf_counter()
        out = d_out.copy_to_host().tobytes()
        d2h_seconds = time.perf_counter() - d2h_t0

        total_seconds = time.perf_counter() - t0
        transfer_seconds = h2d_seconds + d2h_seconds
        return out, GpuTiming(total_seconds, kernel_seconds, transfer_seconds)

    def encrypt_ctr(self, data: bytes, nonce: bytes | None = None) -> tuple[bytes, GpuTiming]:
        """Encrypt `data` in AES-128 CTR mode on the GPU.

        Generates counter blocks on the host, encrypts them via ECB on the GPU
        to produce a keystream, then XORs the keystream with the plaintext.
        Timing reflects total wall time; kernel_seconds tracks only the ECB call.
        """
        nblocks = self._validate_data(data)
        if nblocks == 0:
            return b"", GpuTiming(0.0, 0.0, 0.0)

        ctr_blocks = build_ctr_blocks(nblocks, 16, nonce=nonce)
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
