import numpy as np
from numba import cuda
import os

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"

ROT = np.array([
    (14, 16), (52, 57), (23, 40), (5, 37),
    (25, 33), (46, 12), (58, 22), (32, 32)
], dtype=np.uint64)

C240 = np.uint64(0x1BD11BDAA9FC1A22)


@cuda.jit(device=True)
def rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK


@cuda.jit(device=True)
def rotr(x, n):
    return ((x >> n) | (x << (64 - n))) & MASK


@cuda.jit
def threefish_encrypt_kernel(blocks, k):
    shared_k = cuda.shared.array(5, dtype=np.uint64)

    tid = cuda.threadIdx.x
    if tid < 5:
        shared_k[tid] = k[tid]

    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= blocks.shape[0]:
        return

    x0 = blocks[i, 0]
    x1 = blocks[i, 1]
    x2 = blocks[i, 2]
    x3 = blocks[i, 3]

    for r in range(72):
        if r % 4 == 0:
            s = r // 4
            x0 += shared_k[(s + 0) % 5]
            x1 += shared_k[(s + 1) % 5]
            x2 += shared_k[(s + 2) % 5]
            x3 += shared_k[(s + 3) % 5]

        r0 = ROT[r % 8, 0]
        r1 = ROT[r % 8, 1]

        x0 += x1
        x1 = rotl(x1, r0) ^ x0

        x2 += x3
        x3 = rotl(x3, r1) ^ x2

        tmp = x1
        x1 = x3
        x3 = tmp

    blocks[i, 0] = x0
    blocks[i, 1] = x1
    blocks[i, 2] = x2
    blocks[i, 3] = x3

@cuda.jit
def threefish_decrypt_kernel(blocks, k):
    shared_k = cuda.shared.array(5, dtype=np.uint64)

    tid = cuda.threadIdx.x
    if tid < 5:
        shared_k[tid] = k[tid]

    cuda.syncthreads()

    i = cuda.grid(1)
    if i >= blocks.shape[0]:
        return

    x0 = blocks[i, 0]
    x1 = blocks[i, 1]
    x2 = blocks[i, 2]
    x3 = blocks[i, 3]

    for r in range(71, -1, -1):

        # -----------------------------
        # Inverse permutation
        # -----------------------------
        tmp = x1
        x1 = x3
        x3 = tmp

        r0 = ROT[r % 8, 0]
        r1 = ROT[r % 8, 1]

        # -----------------------------
        # Inverse MIX
        # -----------------------------
        x3 ^= x2
        x3 = rotr(x3, r1)
        x2 = (x2 - x3) & MASK

        x1 ^= x0
        x1 = rotr(x1, r0)
        x0 = (x0 - x1) & MASK

        # -----------------------------
        # Key subtraction
        # -----------------------------
        if r % 4 == 0:
            s = r // 4
            x0 = (x0 - shared_k[(s + 0) % 5]) & MASK
            x1 = (x1 - shared_k[(s + 1) % 5]) & MASK
            x2 = (x2 - shared_k[(s + 2) % 5]) & MASK
            x3 = (x3 - shared_k[(s + 3) % 5]) & MASK

    blocks[i, 0] = x0
    blocks[i, 1] = x1
    blocks[i, 2] = x2
    blocks[i, 3] = x3

# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data: bytes, key: bytes):
    k = np.frombuffer(key, dtype=np.uint64).copy()
    k4 = C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3]
    k = np.append(k, k4)

    pad_len = 32 - (len(data) % 32)
    data += bytes([pad_len]) * pad_len

    blocks = np.frombuffer(data, dtype=np.uint64).reshape(-1, 4).copy()

    d_blocks = cuda.to_device(blocks)
    d_k = cuda.to_device(k)

    threads = 256
    blocks_per_grid = (len(blocks) + threads - 1) // threads

    threefish_encrypt_kernel[blocks_per_grid, threads](d_blocks, d_k)
    cuda.synchronize()

    return d_blocks.copy_to_host().tobytes()

def decrypt(data: bytes, key: bytes):
    assert len(data) % 32 == 0

    k = np.frombuffer(key, dtype=np.uint64).copy()
    k4 = C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3]
    k = np.append(k, k4)

    blocks = np.frombuffer(data, dtype=np.uint64).reshape(-1, 4).copy()

    d_blocks = cuda.to_device(blocks)
    d_k = cuda.to_device(k)

    threads = 256
    blocks_per_grid = (len(blocks) + threads - 1) // threads

    threefish_decrypt_kernel[blocks_per_grid, threads](d_blocks, d_k)
    cuda.synchronize()

    out = d_blocks.copy_to_host().tobytes()

    # Unpad
    pad_len = out[-1]
    assert 1 <= pad_len <= 32
    return out[:-pad_len]