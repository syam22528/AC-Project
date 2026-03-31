MASK = 0xFFFFFFFFFFFFFFFF

ROT = [
    (14, 16), (52, 57), (23, 40), (5, 37),
    (25, 33), (46, 12), (58, 22), (32, 32)
]

def rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK


def mix(x0, x1, r):
    x0 = (x0 + x1) & MASK
    x1 = rotl(x1, r) ^ x0
    return x0, x1


def permute(x):
    return [x[0], x[3], x[2], x[1]]


def key_schedule(key):
    k = [int.from_bytes(key[i:i+8], 'little') for i in range(0, 32, 8)]
    C240 = 0x1BD11BDAA9FC1A22
    k.append(C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3])
    return k


def encrypt_block(block, k):
    x = [int.from_bytes(block[i:i+8], 'little') for i in range(0, 32, 8)]

    for r in range(72):
        if r % 4 == 0:
            for i in range(4):
                x[i] = (x[i] + k[(r//4 + i) % 5]) & MASK

        rc = ROT[r % 8]

        x[0], x[1] = mix(x[0], x[1], rc[0])
        x[2], x[3] = mix(x[2], x[3], rc[1])

        x = permute(x)

    return b''.join(w.to_bytes(8, 'little') for w in x)


def decrypt_block(block, k):
    x = [int.from_bytes(block[i:i+8], 'little') for i in range(0, 32, 8)]

    for r in range(71, -1, -1):
        x = permute(x)

        rc = ROT[r % 8]

        # inverse mix
        x1 = x[1] ^ x[0]
        x1 = ((x1 >> rc[0]) | (x1 << (64 - rc[0]))) & MASK
        x0 = (x[0] - x1) & MASK

        x3 = x[3] ^ x[2]
        x3 = ((x3 >> rc[1]) | (x3 << (64 - rc[1]))) & MASK
        x2 = (x[2] - x3) & MASK

        x = [x0, x1, x2, x3]

        if r % 4 == 0:
            for i in range(4):
                x[i] = (x[i] - k[(r//4 + i) % 5]) & MASK

    return b''.join(w.to_bytes(8, 'little') for w in x)


def pad(data):
    pad_len = 32 - (len(data) % 32)
    return data + bytes([pad_len] * pad_len)


def unpad(data):
    return data[:-data[-1]]

# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data, key):
    k = key_schedule(key)
    data = pad(data)
    return b''.join(encrypt_block(data[i:i+32], k) for i in range(0, len(data), 32))


def decrypt(data, key):
    k = key_schedule(key)
    out = b''.join(decrypt_block(data[i:i+32], k) for i in range(0, len(data), 32))
    return unpad(out)