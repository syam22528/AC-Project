import numpy as np
from numba import njit

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


# -----------------------------
# Helpers
# -----------------------------

@njit
def ROR(x, r):
    return ((x >> r) | (x << (64 - r))) & MASK


@njit
def ROL(x, r):
    return ((x << r) | (x >> (64 - r))) & MASK


# -----------------------------
# Key Schedule
# -----------------------------

@njit
def expand_key(key0, key1):
    round_keys = np.empty(32, dtype=np.uint64)

    k = np.uint64(key0)
    l = np.uint64(key1)

    for i in range(32):
        round_keys[i] = k

        l = (ROR(l, 8) + k) & MASK
        l ^= np.uint64(i)

        k = (ROL(k, 3) ^ l) & MASK

    return round_keys


# -----------------------------
# Core Encryption
# -----------------------------

@njit
def encrypt_blocks(blocks, round_keys):
    n = blocks.shape[0]
    out = np.empty_like(blocks)

    for i in range(n):
        x = blocks[i, 0]
        y = blocks[i, 1]

        for j in range(32):
            k = round_keys[j]

            x = (ROR(x, 8) + y) & MASK
            x ^= k

            y = ROL(y, 3) ^ x
            y &= MASK

        out[i, 0] = x
        out[i, 1] = y

    return out


@njit
def decrypt_blocks(blocks, round_keys):
    n = blocks.shape[0]
    out = np.empty_like(blocks)

    for i in range(n):
        x = blocks[i, 0]
        y = blocks[i, 1]

        for j in range(31, -1, -1):
            k = round_keys[j]

            y ^= x
            y = ROR(y, 3)

            x ^= k
            x = (x - y) & MASK
            x = ROL(x, 8)

        out[i, 0] = x
        out[i, 1] = y

    return out


# -----------------------------
# Helpers (Python side)
# -----------------------------

def bytes_to_blocks(data):
    n = (len(data) + 15) // 16
    blocks = np.zeros((n, 2), dtype=np.uint64)

    for i in range(n):
        chunk = data[i*16:(i+1)*16].ljust(16, b'\x00')
        blocks[i, 0] = int.from_bytes(chunk[:8], 'big')
        blocks[i, 1] = int.from_bytes(chunk[8:], 'big')

    return blocks


def blocks_to_bytes(blocks, original_len):
    out = bytearray()

    for i in range(blocks.shape[0]):
        out.extend(int(blocks[i, 0]).to_bytes(8, 'big'))
        out.extend(int(blocks[i, 1]).to_bytes(8, 'big'))

    return bytes(out[:original_len])


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes):
    key0 = int.from_bytes(key[:8], 'big')
    key1 = int.from_bytes(key[8:], 'big')

    blocks = bytes_to_blocks(data)
    round_keys = expand_key(key0, key1)

    out_blocks = encrypt_blocks(blocks, round_keys)

    # IMPORTANT: return full ciphertext (not truncated)
    return blocks_to_bytes(out_blocks, blocks.shape[0] * 16)


def decrypt(data: bytes, key: bytes):
    key0 = int.from_bytes(key[:8], 'big')
    key1 = int.from_bytes(key[8:], 'big')

    blocks = bytes_to_blocks(data)
    round_keys = expand_key(key0, key1)

    out_blocks = decrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, len(data))