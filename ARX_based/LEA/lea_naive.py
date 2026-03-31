from typing import List

MASK32 = 0xFFFFFFFF


# -----------------------------
# Helpers (32-bit ARX)
# -----------------------------

def rol(x, r):
    return ((x << r) | (x >> (32 - r))) & MASK32


def ror(x, r):
    return ((x >> r) | (x << (32 - r))) & MASK32


def add(x, y):
    return (x + y) & MASK32


def sub(x, y):
    return (x - y) & MASK32


# -----------------------------
# Constants (δ)
# -----------------------------
DELTA = [
    0xc3efe9db,
    0x44626b02,
    0x79e27c8a,
    0x78df30ec
]


# -----------------------------
# Key Schedule (LEA-128)
# -----------------------------

def expand_key(key: bytes) -> List[List[int]]:
    assert len(key) == 16

    T = [
        int.from_bytes(key[i*4:(i+1)*4], 'big')
        for i in range(4)
    ]

    round_keys = []

    for i in range(24):
        d = DELTA[i % 4]

        T[0] = rol(add(T[0], rol(d, i)), 1)
        T[1] = rol(add(T[1], rol(d, i + 1)), 3)
        T[2] = rol(add(T[2], rol(d, i + 2)), 6)
        T[3] = rol(add(T[3], rol(d, i + 3)), 11)

        # Ki = T0 || T1 || T2 || T1 || T3 || T1
        round_keys.append([
            T[0], T[1], T[2],
            T[1], T[3], T[1]
        ])

    return round_keys


# -----------------------------
# Encrypt single block
# -----------------------------

def encrypt_block(block: bytes, round_keys):
    assert len(block) == 16

    X = [
        int.from_bytes(block[i*4:(i+1)*4], 'big')
        for i in range(4)
    ]

    for i in range(24):
        k = round_keys[i]

        x0 = rol(add(X[0] ^ k[0], X[1] ^ k[1]), 9)
        x1 = ror(add(X[1] ^ k[2], X[2] ^ k[3]), 5)
        x2 = ror(add(X[2] ^ k[4], X[3] ^ k[5]), 3)
        x3 = X[0]

        X = [x0, x1, x2, x3]

    out = bytearray()
    for v in X:
        out.extend(v.to_bytes(4, 'big'))

    return bytes(out)


# -----------------------------
# Decrypt single block
# -----------------------------

def decrypt_block(block: bytes, round_keys):
    assert len(block) == 16

    X = [
        int.from_bytes(block[i*4:(i+1)*4], 'big')
        for i in range(4)
    ]

    for i in reversed(range(24)):
        k = round_keys[i]

        x_next0, x_next1, x_next2, x_next3 = X

        # Xi[0]
        x0 = x_next3 & MASK32

        # Xi[1]
        t0 = ror(x_next0, 9)
        t0 = sub(t0, (x0 ^ k[0]))
        x1 = (t0 ^ k[1]) & MASK32

        # Xi[2]
        t1 = rol(x_next1, 5)
        t1 = sub(t1, (x1 ^ k[2]))
        x2 = (t1 ^ k[3]) & MASK32

        # Xi[3]
        t2 = rol(x_next2, 3)
        t2 = sub(t2, (x2 ^ k[4]))
        x3 = (t2 ^ k[5]) & MASK32

        X = [x0, x1, x2, x3]

    out = bytearray()
    for v in X:
        out.extend(v.to_bytes(4, 'big'))

    return bytes(out)

# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes) -> bytes:
    round_keys = expand_key(key)

    out = bytearray()

    for i in range(0, len(data), 16):
        block = data[i:i+16].ljust(16, b'\x00')
        out.extend(encrypt_block(block, round_keys))

    return bytes(out)


def decrypt(data: bytes, key: bytes) -> bytes:
    round_keys = expand_key(key)

    out = bytearray()

    for i in range(0, len(data), 16):
        block = data[i:i+16]
        out.extend(decrypt_block(block, round_keys))

    return bytes(out).rstrip(b'\x00')   