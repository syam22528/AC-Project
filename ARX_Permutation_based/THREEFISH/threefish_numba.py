import numpy as np
from numba import njit

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

ROT = np.array([
    (14, 16), (52, 57), (23, 40), (5, 37),
    (25, 33), (46, 12), (58, 22), (32, 32)
], dtype=np.uint64)

C240 = np.uint64(0x1BD11BDAA9FC1A22)


@njit
def rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK


@njit
def rotr(x, n):
    return ((x >> n) | (x << (64 - n))) & MASK


@njit
def key_schedule(k):
    k4 = C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3]
    out = np.empty(5, dtype=np.uint64)
    out[0] = k[0]
    out[1] = k[1]
    out[2] = k[2]
    out[3] = k[3]
    out[4] = k4
    return out


@njit
def encrypt_blocks(blocks, k):
    n = blocks.shape[0]

    for b in range(n):
        x0 = blocks[b, 0]
        x1 = blocks[b, 1]
        x2 = blocks[b, 2]
        x3 = blocks[b, 3]

        for r in range(72):
            if r % 4 == 0:
                s = r // 4
                x0 = (x0 + k[(s + 0) % 5]) & MASK
                x1 = (x1 + k[(s + 1) % 5]) & MASK
                x2 = (x2 + k[(s + 2) % 5]) & MASK
                x3 = (x3 + k[(s + 3) % 5]) & MASK

            r0 = ROT[r % 8, 0]
            r1 = ROT[r % 8, 1]

            x0 = (x0 + x1) & MASK
            x1 = rotl(x1, r0) ^ x0

            x2 = (x2 + x3) & MASK
            x3 = rotl(x3, r1) ^ x2

            # permute
            tmp = x1
            x1 = x3
            x3 = tmp

        blocks[b, 0] = x0
        blocks[b, 1] = x1
        blocks[b, 2] = x2
        blocks[b, 3] = x3


@njit
def decrypt_blocks(blocks, k):
    n = blocks.shape[0]

    for b in range(n):
        x0 = blocks[b, 0]
        x1 = blocks[b, 1]
        x2 = blocks[b, 2]
        x3 = blocks[b, 3]

        for r in range(71, -1, -1):
            # inverse permute
            tmp = x1
            x1 = x3
            x3 = tmp

            r0 = ROT[r % 8, 0]
            r1 = ROT[r % 8, 1]

            x3 ^= x2
            x3 = rotr(x3, r1)
            x2 = (x2 - x3) & MASK

            x1 ^= x0
            x1 = rotr(x1, r0)
            x0 = (x0 - x1) & MASK

            if r % 4 == 0:
                s = r // 4
                x0 = (x0 - k[(s + 0) % 5]) & MASK
                x1 = (x1 - k[(s + 1) % 5]) & MASK
                x2 = (x2 - k[(s + 2) % 5]) & MASK
                x3 = (x3 - k[(s + 3) % 5]) & MASK

        blocks[b, 0] = x0
        blocks[b, 1] = x1
        blocks[b, 2] = x2
        blocks[b, 3] = x3


# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data: bytes, key: bytes):
    k = np.frombuffer(key, dtype=np.uint64).copy()
    k = key_schedule(k)

    pad_len = 32 - (len(data) % 32)
    data += bytes([pad_len]) * pad_len

    arr = np.frombuffer(data, dtype=np.uint64).reshape(-1, 4).copy()

    encrypt_blocks(arr, k)

    return arr.tobytes()


def decrypt(data: bytes, key: bytes):
    k = np.frombuffer(key, dtype=np.uint64).copy()
    k = key_schedule(k)

    arr = np.frombuffer(data, dtype=np.uint64).reshape(-1, 4).copy()

    decrypt_blocks(arr, k)

    out = arr.tobytes()
    pad_len = out[-1]
    return out[:-pad_len]