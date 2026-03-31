import numpy as np
from numba import njit

MASK32 = np.uint32(0xFFFFFFFF)


# -----------------------------
# Helpers
# -----------------------------

@njit
def rol(x, r):
    return ((x << r) | (x >> (32 - r))) & MASK32


@njit
def ror(x, r):
    return ((x >> r) | (x << (32 - r))) & MASK32


# -----------------------------
# Constants
# -----------------------------

DELTA = np.array([
    0xc3efe9db,
    0x44626b02,
    0x79e27c8a,
    0x78df30ec
], dtype=np.uint32)


# -----------------------------
# Key Schedule
# -----------------------------

@njit
def expand_key(key_words):
    T = key_words.copy()
    round_keys = np.zeros((24, 6), dtype=np.uint32)

    for i in range(24):
        d = DELTA[i % 4]

        T[0] = rol((T[0] + rol(d, i)) & MASK32, 1)
        T[1] = rol((T[1] + rol(d, i + 1)) & MASK32, 3)
        T[2] = rol((T[2] + rol(d, i + 2)) & MASK32, 6)
        T[3] = rol((T[3] + rol(d, i + 3)) & MASK32, 11)

        round_keys[i, 0] = T[0]
        round_keys[i, 1] = T[1]
        round_keys[i, 2] = T[2]
        round_keys[i, 3] = T[1]
        round_keys[i, 4] = T[3]
        round_keys[i, 5] = T[1]

    return round_keys


# -----------------------------
# Encrypt Blocks
# -----------------------------
@njit
def encrypt_blocks(blocks, round_keys):
    n = blocks.shape[0]
    out = np.empty_like(blocks)

    for i in range(n):
        x0 = blocks[i, 0]
        x1 = blocks[i, 1]
        x2 = blocks[i, 2]
        x3 = blocks[i, 3]

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

        out[i, 0] = x0
        out[i, 1] = x1
        out[i, 2] = x2
        out[i, 3] = x3

    return out


# -----------------------------
# Decrypt Blocks
# -----------------------------
@njit
def decrypt_blocks(blocks, round_keys):
    n = blocks.shape[0]
    out = np.empty_like(blocks)

    for i in range(n):
        x0 = blocks[i, 0]
        x1 = blocks[i, 1]
        x2 = blocks[i, 2]
        x3 = blocks[i, 3]

        for r in range(23, -1, -1):
            k0 = round_keys[r, 0]
            k1 = round_keys[r, 1]
            k2 = round_keys[r, 2]
            k3 = round_keys[r, 3]
            k4 = round_keys[r, 4]
            k5 = round_keys[r, 5]

            # Save current state (Xi+1)
            x_next0 = x0
            x_next1 = x1
            x_next2 = x2
            x_next3 = x3

            # Correct dependency order
            x0 = x_next3

            t0 = ror(x_next0, 9)
            x1 = (t0 - (x0 ^ k0)) ^ k1

            t1 = rol(x_next1, 5)
            x2 = (t1 - (x1 ^ k2)) ^ k3

            t2 = rol(x_next2, 3)
            x3 = (t2 - (x2 ^ k4)) ^ k5

            x0 &= MASK32
            x1 &= MASK32
            x2 &= MASK32
            x3 &= MASK32

        out[i, 0] = x0
        out[i, 1] = x1
        out[i, 2] = x2
        out[i, 3] = x3

    return out


# -----------------------------
# Python-side helpers
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
    key_words = np.array([
        int.from_bytes(key[i*4:(i+1)*4], 'big')
        for i in range(4)
    ], dtype=np.uint32)

    blocks = bytes_to_blocks(data)

    round_keys = expand_key(key_words)
    out_blocks = encrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, out_blocks.shape[0] * 16)


def decrypt(data: bytes, key: bytes):
    key_words = np.array([
        int.from_bytes(key[i*4:(i+1)*4], 'big')
        for i in range(4)
    ], dtype=np.uint32)

    blocks = bytes_to_blocks(data)

    round_keys = expand_key(key_words)
    out_blocks = decrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, out_blocks.shape[0] * 16).rstrip(b'\x00')