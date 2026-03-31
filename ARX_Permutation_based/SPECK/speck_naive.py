MASK = 0xFFFFFFFFFFFFFFFF


def ROR(x, r):
    return ((x >> r) | (x << (64 - r))) & MASK


def ROL(x, r):
    return ((x << r) | (x >> (64 - r))) & MASK


# -----------------------------
# Key Schedule (SINGLE SOURCE OF TRUTH)
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
# Encrypt block
# -----------------------------

def encrypt_block(x, y, round_keys):
    for k in round_keys:
        x = (ROR(x, 8) + y) & MASK
        x ^= k
        y = ROL(y, 3) ^ x
        y &= MASK
    return x, y


# -----------------------------
# Decrypt block (EXACT INVERSE)
# -----------------------------

def decrypt_block(x, y, round_keys):
    for k in reversed(round_keys):
        y ^= x
        y = ROR(y, 3)

        x ^= k
        x = (x - y) & MASK
        x = ROL(x, 8)

    return x, y


# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data, key):
    round_keys = expand_key(key)
    out = bytearray()

    for i in range(0, len(data), 16):
        block = data[i:i+16].ljust(16, b'\x00')

        x = int.from_bytes(block[:8], 'big')
        y = int.from_bytes(block[8:], 'big')

        x, y = encrypt_block(x, y, round_keys)

        out.extend(x.to_bytes(8, 'big'))
        out.extend(y.to_bytes(8, 'big'))

    return bytes(out)


def decrypt(data, key):
    round_keys = expand_key(key)
    out = bytearray()

    for i in range(0, len(data), 16):
        block = data[i:i+16]

        x = int.from_bytes(block[:8], 'big')
        y = int.from_bytes(block[8:], 'big')

        x, y = decrypt_block(x, y, round_keys)

        out.extend(x.to_bytes(8, 'big'))
        out.extend(y.to_bytes(8, 'big'))

    return bytes(out)