import numpy as np
from numba import cuda
import os

MASK = np.uint32(0xFFFFFFFF)
os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"

# -----------------------------
# Device functions
# -----------------------------
@cuda.jit(device=True)
def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & MASK


@cuda.jit(device=True)
def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


# -----------------------------
# KERNELS 
# -----------------------------
@cuda.jit
def cham_encrypt_kernel(blocks, rk):
    # Shared memory for round keys
    shared_rk = cuda.shared.array(16, dtype=np.uint32)

    tid = cuda.threadIdx.x

    # Load round keys into shared memory
    if tid < 16:
        shared_rk[tid] = rk[tid]

    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= blocks.shape[0]:
        return

    x0 = blocks[i, 0]
    x1 = blocks[i, 1]
    x2 = blocks[i, 2]
    x3 = blocks[i, 3]

    # 80 rounds
    for r in range(80):
        rk_i = shared_rk[r & 15]

        if r & 1:
            t = rotl(x0, 8) ^ x2 ^ (x1 + rk_i)
        else:
            t = rotl(x0, 1) ^ x2 ^ (x1 + rk_i)

        x0, x1, x2, x3 = x1, x2, x3, t & MASK

    blocks[i, 0] = x0
    blocks[i, 1] = x1
    blocks[i, 2] = x2
    blocks[i, 3] = x3

@cuda.jit
def cham_decrypt_kernel(blocks, rk):
    shared_rk = cuda.shared.array(16, dtype=np.uint32)

    tid = cuda.threadIdx.x

    if tid < 16:
        shared_rk[tid] = rk[tid]

    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= blocks.shape[0]:
        return

    x0 = blocks[i, 0]
    x1 = blocks[i, 1]
    x2 = blocks[i, 2]
    x3 = blocks[i, 3]

    for r in range(79, -1, -1):
        rk_i = shared_rk[r & 15]

        inner = x3 ^ x1 ^ (x0 + rk_i)

        if r & 1:
            old_x0 = rotr(inner, 8)
        else:
            old_x0 = rotr(inner, 1)

        x0, x1, x2, x3 = old_x0 & MASK, x0, x1, x2

    blocks[i, 0] = x0
    blocks[i, 1] = x1
    blocks[i, 2] = x2
    blocks[i, 3] = x3

# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data: bytes, key: bytes):
    from CHAM.cham_optimized import key_schedule

    rk = np.array(key_schedule(key), dtype=np.uint32)

    # Padding
    pad_len = 16 - (len(data) % 16)
    data += bytes([pad_len]) * pad_len

    blocks = np.frombuffer(data, dtype=np.uint32).reshape(-1, 4).copy()

    # GPU transfer
    d_blocks = cuda.to_device(blocks)
    d_rk = cuda.to_device(rk)

    threads = 256
    blocks_per_grid = (len(blocks) + threads - 1) // threads

    cham_encrypt_kernel[blocks_per_grid, threads](d_blocks, d_rk)
    cuda.synchronize()

    return d_blocks.copy_to_host().astype(np.uint32).tobytes()


def decrypt(data: bytes, key: bytes):
    from CHAM.cham_optimized import key_schedule

    assert len(data) % 16 == 0

    rk = np.array(key_schedule(key), dtype=np.uint32)

    blocks = np.frombuffer(data, dtype=np.uint32).reshape(-1, 4).copy()

    d_blocks = cuda.to_device(blocks)
    d_rk = cuda.to_device(rk)

    threads = 256
    blocks_per_grid = (len(blocks) + threads - 1) // threads

    cham_decrypt_kernel[blocks_per_grid, threads](d_blocks, d_rk)
    cuda.synchronize()

    out = d_blocks.copy_to_host().astype(np.uint32).tobytes()

    # Unpad
    pad_len = out[-1]
    assert 1 <= pad_len <= 16
    return out[:-pad_len]