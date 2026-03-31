from numba import cuda
import numpy as np
import os

# CUDA setup (Windows fix)
os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"

MASK32 = np.uint32(0xFFFFFFFF)


# -----------------------------
# Device Helpers
# -----------------------------

@cuda.jit(device=True, inline=True)
def rol(x, r):
    return ((x << r) | (x >> (32 - r))) & MASK32


@cuda.jit(device=True, inline=True)
def ror(x, r):
    return ((x >> r) | (x << (32 - r))) & MASK32


# -----------------------------
# Kernel
# -----------------------------

@cuda.jit
def lea_kernel(blocks, out, round_keys):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)

    for j in range(i, blocks.shape[0], stride):

        x0 = blocks[j, 0]
        x1 = blocks[j, 1]
        x2 = blocks[j, 2]
        x3 = blocks[j, 3]

        for r in range(24):
            k0 = round_keys[r, 0]
            k1 = round_keys[r, 1]
            k2 = round_keys[r, 2]
            k3 = round_keys[r, 3]
            k4 = round_keys[r, 4]
            k5 = round_keys[r, 5]

            tmp0 = ((x0 ^ k0) + (x1 ^ k1)) & MASK32
            tmp1 = ((x1 ^ k2) + (x2 ^ k3)) & MASK32
            tmp2 = ((x2 ^ k4) + (x3 ^ k5)) & MASK32

            new_x0 = rol(tmp0, 9)
            new_x1 = ror(tmp1, 5)
            new_x2 = ror(tmp2, 3)
            new_x3 = x0

            x0 = new_x0
            x1 = new_x1
            x2 = new_x2
            x3 = new_x3

        out[j, 0] = x0
        out[j, 1] = x1
        out[j, 2] = x2
        out[j, 3] = x3


# -----------------------------
# Key Schedule (CPU)
# -----------------------------

def expand_key(key: bytes):
    DELTA = [
        0xc3efe9db,
        0x44626b02,
        0x79e27c8a,
        0x78df30ec
    ]

    T = [
        int.from_bytes(key[i*4:(i+1)*4], 'big')
        for i in range(4)
    ]

    round_keys = np.zeros((24, 6), dtype=np.uint32)

    for i in range(24):
        d = DELTA[i % 4]

        def rol_cpu(x, r):
            return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF

        T[0] = rol_cpu((T[0] + rol_cpu(d, i)) & 0xFFFFFFFF, 1)
        T[1] = rol_cpu((T[1] + rol_cpu(d, i+1)) & 0xFFFFFFFF, 3)
        T[2] = rol_cpu((T[2] + rol_cpu(d, i+2)) & 0xFFFFFFFF, 6)
        T[3] = rol_cpu((T[3] + rol_cpu(d, i+3)) & 0xFFFFFFFF, 11)

        round_keys[i] = [T[0], T[1], T[2], T[1], T[3], T[1]]

    return round_keys


# -----------------------------
# Helpers
# -----------------------------

def bytes_to_blocks(data):
    n = (len(data) + 15) // 16
    blocks = np.zeros((n, 4), dtype=np.uint32)

    for i in range(n):
        chunk = data[i*16:(i+1)*16].ljust(16, b'\x00')
        for j in range(4):
            blocks[i, j] = int.from_bytes(chunk[j*4:(j+1)*4], 'big')

    return blocks


def blocks_to_bytes(blocks, length):
    out = bytearray()

    for i in range(blocks.shape[0]):
        for j in range(4):
            out.extend(int(blocks[i, j]).to_bytes(4, 'big'))

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
    blocks_per_grid = (blocks.shape[0] + threads_per_block - 1) // threads_per_block

    lea_kernel[blocks_per_grid, threads_per_block](d_blocks, d_out, d_keys)

    result = d_out.copy_to_host()

    return blocks_to_bytes(result, blocks.shape[0] * 16)

