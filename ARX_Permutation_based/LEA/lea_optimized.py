import numpy as np
import numpy as np
np.seterr(over='ignore')

MASK32 = np.uint32(0xFFFFFFFF)


# -----------------------------
# Helpers (vector-friendly)
# -----------------------------

def rol(x, r):
    return ((x << r) | (x >> (32 - r))) & MASK32


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

def expand_key(key: bytes):
    T = np.array([
        int.from_bytes(key[i*4:(i+1)*4], 'big')
        for i in range(4)
    ], dtype=np.uint32)

    round_keys = np.zeros((24, 6), dtype=np.uint32)

    for i in range(24):
        d = DELTA[i % 4]

        T[0] = rol((T[0] + rol(d, i)) & MASK32, 1)
        T[1] = rol((T[1] + rol(d, i+1)) & MASK32, 3)
        T[2] = rol((T[2] + rol(d, i+2)) & MASK32, 6)
        T[3] = rol((T[3] + rol(d, i+3)) & MASK32, 11)

        round_keys[i] = [T[0], T[1], T[2], T[1], T[3], T[1]]

    return round_keys


# -----------------------------
# Block Conversion
# -----------------------------

def bytes_to_blocks(data: bytes):
    n = (len(data) + 15) // 16
    blocks = np.zeros((n, 4), dtype=np.uint32)

    for i in range(n):
        chunk = data[i*16:(i+1)*16].ljust(16, b'\x00')
        for j in range(4):
            blocks[i, j] = int.from_bytes(chunk[j*4:(j+1)*4], 'big')

    return blocks


def blocks_to_bytes(blocks, original_len):
    out = bytearray()

    for row in blocks:
        for val in row:
            out.extend(int(val).to_bytes(4, 'big'))

    return bytes(out[:original_len])


# -----------------------------
# Encrypt Blocks (vectorized loop)
# -----------------------------

def encrypt_blocks(blocks, round_keys):
    X = blocks.copy()

    for i in range(24):
        k = round_keys[i]

        x0_old = X[:, 0].copy()
        x1_old = X[:, 1].copy()
        x2_old = X[:, 2].copy()
        x3_old = X[:, 3].copy()

        x0 = rol((x0_old ^ k[0]) + (x1_old ^ k[1]), 9)
        x1 = ror((x1_old ^ k[2]) + (x2_old ^ k[3]), 5)
        x2 = ror((x2_old ^ k[4]) + (x3_old ^ k[5]), 3)
        x3 = x0_old   # correct dependency

        X[:, 0] = x0 & MASK32
        X[:, 1] = x1 & MASK32
        X[:, 2] = x2 & MASK32
        X[:, 3] = x3 & MASK32

    return X


# -----------------------------
# Decrypt Blocks
# -----------------------------

def decrypt_blocks(blocks, round_keys):
    X = blocks.copy()

    for i in range(23, -1, -1):
        k = round_keys[i]

        x_next0 = X[:, 0].copy()
        x_next1 = X[:, 1].copy()
        x_next2 = X[:, 2].copy()
        x_next3 = X[:, 3].copy()

        x0 = x_next3

        t0 = ror(x_next0, 9)
        x1 = (t0 - (x0 ^ k[0])) ^ k[1]

        t1 = rol(x_next1, 5)
        x2 = (t1 - (x1 ^ k[2])) ^ k[3]

        t2 = rol(x_next2, 3)
        x3 = (t2 - (x2 ^ k[4])) ^ k[5]

        X[:, 0] = x0 & MASK32
        X[:, 1] = x1 & MASK32
        X[:, 2] = x2 & MASK32
        X[:, 3] = x3 & MASK32

    return X


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes):
    round_keys = expand_key(key)
    blocks = bytes_to_blocks(data)

    out_blocks = encrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, out_blocks.shape[0] * 16)


def decrypt(data: bytes, key: bytes):
    round_keys = expand_key(key)
    blocks = bytes_to_blocks(data)

    out_blocks = decrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, len(data)).rstrip(b'\x00')