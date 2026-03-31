from numba import cuda
import numpy as np
import os

os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


# -----------------------------
# Device Helpers
# -----------------------------

@cuda.jit(device=True, inline=True)
def ROR(x, r):
    return ((x >> r) | (x << (64 - r))) & MASK


@cuda.jit(device=True, inline=True)
def ROL(x, r):
    return ((x << r) | (x >> (64 - r))) & MASK


# -----------------------------
# Optimized Kernel
# -----------------------------

@cuda.jit
def speck_kernel(blocks, out, round_keys, decrypt):
    # Shared memory for round keys (FAST)
    shared_keys = cuda.shared.array(32, dtype=np.uint64)

    tid = cuda.threadIdx.x

    # Load keys into shared memory
    if tid < 32:
        shared_keys[tid] = round_keys[tid]

    cuda.syncthreads()

    # Grid-stride loop
    i = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(i, blocks.shape[0], stride):

        x = blocks[j, 0]
        y = blocks[j, 1]

        if decrypt == 0:
            # ENCRYPT
            for r in range(32):
                k = shared_keys[r]

                x = (ROR(x, 8) + y) & MASK
                x ^= k

                y = ROL(y, 3) ^ x
                y &= MASK

        else:
            # DECRYPT
            for r in range(31, -1, -1):
                k = shared_keys[r]

                y ^= x
                y = ROR(y, 3)

                x ^= k
                x = (x - y) & MASK
                x = ROL(x, 8)

        out[j, 0] = x
        out[j, 1] = y


# -----------------------------
# Key Schedule (CPU)
# -----------------------------

def expand_key(key):
    k = int.from_bytes(key[:8], 'big')
    l = int.from_bytes(key[8:], 'big')

    round_keys = np.zeros(32, dtype=np.uint64)

    for i in range(32):
        round_keys[i] = k

        l = ((l >> 8) | (l << (64 - 8))) & 0xFFFFFFFFFFFFFFFF
        l = (l + k) & 0xFFFFFFFFFFFFFFFF
        l ^= i

        k = ((k << 3) | (k >> (64 - 3))) & 0xFFFFFFFFFFFFFFFF
        k ^= l

    return round_keys


# -----------------------------
# Helpers
# -----------------------------

def bytes_to_blocks(data):
    n = (len(data) + 15) // 16
    blocks = np.zeros((n, 2), dtype=np.uint64)

    for i in range(n):
        chunk = data[i*16:(i+1)*16].ljust(16, b'\x00')
        blocks[i, 0] = int.from_bytes(chunk[:8], 'big')
        blocks[i, 1] = int.from_bytes(chunk[8:], 'big')

    return blocks


def blocks_to_bytes(blocks, length):
    out = bytearray()

    for i in range(blocks.shape[0]):
        out.extend(int(blocks[i, 0]).to_bytes(8, 'big'))
        out.extend(int(blocks[i, 1]).to_bytes(8, 'big'))

    return bytes(out[:length])


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes):
    blocks = bytes_to_blocks(data)
    round_keys = expand_key(key)

    d_blocks = cuda.to_device(blocks)
    d_out = cuda.device_array_like(blocks)
    d_keys = cuda.to_device(round_keys)

    threads_per_block = 256

    # Better occupancy
    blocks_per_grid = min(65535, (blocks.shape[0] + threads_per_block - 1) // threads_per_block)

    speck_kernel[blocks_per_grid, threads_per_block](
        d_blocks, d_out, d_keys, 0
    )

    result = d_out.copy_to_host()

    return blocks_to_bytes(result, blocks.shape[0] * 16)


def decrypt(data: bytes, key: bytes):
    blocks = bytes_to_blocks(data)
    round_keys = expand_key(key)

    d_blocks = cuda.to_device(blocks)
    d_out = cuda.device_array_like(blocks)
    d_keys = cuda.to_device(round_keys)

    threads_per_block = 256
    blocks_per_grid = min(65535, (blocks.shape[0] + threads_per_block - 1) // threads_per_block)

    speck_kernel[blocks_per_grid, threads_per_block](
        d_blocks, d_out, d_keys, 1
    )

    result = d_out.copy_to_host()

    return blocks_to_bytes(result, len(data))