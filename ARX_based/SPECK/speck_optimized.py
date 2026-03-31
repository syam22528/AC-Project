# SPECK/speck_optimized.py

MASK = 0xFFFFFFFFFFFFFFFF


# -----------------------------
# Helpers
# -----------------------------

def ROR(x, r):
    return ((x >> r) | (x << (64 - r))) & MASK


def ROL(x, r):
    return ((x << r) | (x >> (64 - r))) & MASK


# -----------------------------
# Key Schedule
# -----------------------------

def expand_key(key):
    k = int.from_bytes(key[:8], 'big')
    l = int.from_bytes(key[8:], 'big')

    round_keys = []

    for i in range(32):
        round_keys.append(k)

        l = (ROR(l, 8) + k) & MASK
        l ^= i

        k = ROL(k, 3) ^ l
        k &= MASK

    return round_keys


# -----------------------------
# Block Processing (FIXED)
# -----------------------------

def encrypt_blocks(blocks, round_keys):
    out = []

    for x, y in blocks:
        for k in round_keys:
            x = (ROR(x, 8) + y) & MASK
            x ^= k
            y = ROL(y, 3) ^ x
            y &= MASK

        out.append((x, y))

    return out


def decrypt_blocks(blocks, round_keys):
    out = []

    for x, y in blocks:
        for k in reversed(round_keys):
            y ^= x
            y = ROR(y, 3)

            x ^= k
            x = (x - y) & MASK
            x = ROL(x, 8)

        out.append((x, y))

    return out


# -----------------------------
# Helpers (NO NUMPY)
# -----------------------------

def bytes_to_blocks(data):
    n = (len(data) + 15) // 16
    blocks = []

    for i in range(n):
        chunk = data[i*16:(i+1)*16].ljust(16, b'\x00')
        x = int.from_bytes(chunk[:8], 'big')
        y = int.from_bytes(chunk[8:], 'big')
        blocks.append((x, y))

    return blocks


def blocks_to_bytes(blocks, length):
    out = bytearray()

    for x, y in blocks:
        out.extend(x.to_bytes(8, 'big'))
        out.extend(y.to_bytes(8, 'big'))

    return bytes(out[:length])


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes):
    round_keys = expand_key(key)
    blocks = bytes_to_blocks(data)

    out_blocks = encrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, len(out_blocks) * 16)

def decrypt(data: bytes, key: bytes):
    round_keys = expand_key(key)
    blocks = bytes_to_blocks(data)

    out_blocks = decrypt_blocks(blocks, round_keys)

    return blocks_to_bytes(out_blocks, len(data))