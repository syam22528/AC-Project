import numpy as np
from numba import njit

MASK = np.uint32(0xFFFFFFFF)

@njit
def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & MASK

@njit
def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


@njit
def encrypt_blocks(blocks, rk):
    n = blocks.shape[0]

    for b in range(n):
        x0 = blocks[b, 0]
        x1 = blocks[b, 1]
        x2 = blocks[b, 2]
        x3 = blocks[b, 3]

        for i in range(80):
            rk_i = rk[i & 15]

            if i & 1:
                t = (rotl(x0, 8) ^ x2 ^ ((x1 + rk_i) & MASK)) & MASK
            else:
                t = (rotl(x0, 1) ^ x2 ^ ((x1 + rk_i) & MASK)) & MASK

            x0, x1, x2, x3 = x1, x2, x3, t

        blocks[b, 0] = x0
        blocks[b, 1] = x1
        blocks[b, 2] = x2
        blocks[b, 3] = x3


@njit
def decrypt_blocks(blocks, rk):
    n = blocks.shape[0]

    for b in range(n):
        x0 = blocks[b, 0]
        x1 = blocks[b, 1]
        x2 = blocks[b, 2]
        x3 = blocks[b, 3]

        for i in range(79, -1, -1):
            rk_i = rk[i & 15]

            inner = (x3 ^ x1 ^ ((x0 + rk_i) & MASK)) & MASK

            if i & 1:
                old_x0 = rotr(inner, 8)
            else:
                old_x0 = rotr(inner, 1)

            x0, x1, x2, x3 = old_x0, x0, x1, x2

        blocks[b, 0] = x0
        blocks[b, 1] = x1
        blocks[b, 2] = x2
        blocks[b, 3] = x3


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes):
    from CHAM.cham_optimized import key_schedule

    rk = np.array(key_schedule(key), dtype=np.uint32)

    pad_len = 16 - (len(data) % 16)
    data += bytes([pad_len]) * pad_len

    arr = np.frombuffer(data, dtype=np.uint32).reshape(-1, 4).copy()

    encrypt_blocks(arr, rk)

    return arr.astype(np.uint32).tobytes()

def decrypt(data: bytes, key: bytes):
    from CHAM.cham_optimized import key_schedule

    assert len(data) % 16 == 0

    rk = np.array(key_schedule(key), dtype=np.uint32)

    arr = np.frombuffer(data, dtype=np.uint32).reshape(-1, 4).copy()

    decrypt_blocks(arr, rk)

    out = arr.astype(np.uint32).tobytes()

    # unpad
    pad_len = out[-1]
    return out[:-pad_len]