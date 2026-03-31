MASK = 0xFFFFFFFFFFFFFFFF

ROT = [
    (14, 16), (52, 57), (23, 40), (5, 37),
    (25, 33), (46, 12), (58, 22), (32, 32)
]

C240 = 0x1BD11BDAA9FC1A22


def rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK

def rotr(x, n):
    return ((x >> n) | (x << (64 - n))) & MASK


def key_schedule(key: bytes):
    k0 = int.from_bytes(key[0:8], 'little')
    k1 = int.from_bytes(key[8:16], 'little')
    k2 = int.from_bytes(key[16:24], 'little')
    k3 = int.from_bytes(key[24:32], 'little')

    k4 = C240 ^ k0 ^ k1 ^ k2 ^ k3

    return (k0, k1, k2, k3, k4)


# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data: bytes, key: bytes) -> bytes:
    k0, k1, k2, k3, k4 = key_schedule(key)

    # padding
    pad_len = 32 - (len(data) % 32)
    data += bytes([pad_len]) * pad_len

    out = bytearray(len(data))

    for off in range(0, len(data), 32):
        x0 = int.from_bytes(data[off:off+8], 'little')
        x1 = int.from_bytes(data[off+8:off+16], 'little')
        x2 = int.from_bytes(data[off+16:off+24], 'little')
        x3 = int.from_bytes(data[off+24:off+32], 'little')

        for r in range(72):
            if r % 4 == 0:
                s = r // 4
                x0 = (x0 + (k0, k1, k2, k3, k4)[(s + 0) % 5]) & MASK
                x1 = (x1 + (k0, k1, k2, k3, k4)[(s + 1) % 5]) & MASK
                x2 = (x2 + (k0, k1, k2, k3, k4)[(s + 2) % 5]) & MASK
                x3 = (x3 + (k0, k1, k2, k3, k4)[(s + 3) % 5]) & MASK

            r0, r1 = ROT[r % 8]

            x0 = (x0 + x1) & MASK
            x1 = rotl(x1, r0) ^ x0

            x2 = (x2 + x3) & MASK
            x3 = rotl(x3, r1) ^ x2

            # permute
            x1, x3 = x3, x1

        out[off:off+8] = x0.to_bytes(8, 'little')
        out[off+8:off+16] = x1.to_bytes(8, 'little')
        out[off+16:off+24] = x2.to_bytes(8, 'little')
        out[off+24:off+32] = x3.to_bytes(8, 'little')

    return bytes(out)


def decrypt(data: bytes, key: bytes) -> bytes:
    k0, k1, k2, k3, k4 = key_schedule(key)

    out = bytearray(len(data))

    for off in range(0, len(data), 32):
        x0 = int.from_bytes(data[off:off+8], 'little')
        x1 = int.from_bytes(data[off+8:off+16], 'little')
        x2 = int.from_bytes(data[off+16:off+24], 'little')
        x3 = int.from_bytes(data[off+24:off+32], 'little')

        for r in range(71, -1, -1):
            # inverse permute
            x1, x3 = x3, x1

            r0, r1 = ROT[r % 8]

            # inverse mix
            x3 ^= x2
            x3 = rotr(x3, r1)
            x2 = (x2 - x3) & MASK

            x1 ^= x0
            x1 = rotr(x1, r0)
            x0 = (x0 - x1) & MASK

            if r % 4 == 0:
                s = r // 4
                x0 = (x0 - (k0, k1, k2, k3, k4)[(s + 0) % 5]) & MASK
                x1 = (x1 - (k0, k1, k2, k3, k4)[(s + 1) % 5]) & MASK
                x2 = (x2 - (k0, k1, k2, k3, k4)[(s + 2) % 5]) & MASK
                x3 = (x3 - (k0, k1, k2, k3, k4)[(s + 3) % 5]) & MASK

        out[off:off+8] = x0.to_bytes(8, 'little')
        out[off+8:off+16] = x1.to_bytes(8, 'little')
        out[off+16:off+24] = x2.to_bytes(8, 'little')
        out[off+24:off+32] = x3.to_bytes(8, 'little')

    pad_len = out[-1]
    return bytes(out[:-pad_len])