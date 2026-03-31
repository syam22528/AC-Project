# CHAM/cham_optimized.py

MASK = 0xFFFFFFFF

def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & MASK

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def key_schedule(key: bytes):
    k0 = int.from_bytes(key[0:4], 'little')
    k1 = int.from_bytes(key[4:8], 'little')
    k2 = int.from_bytes(key[8:12], 'little')
    k3 = int.from_bytes(key[12:16], 'little')

    rk = [0] * 16

    k = [k0, k1, k2, k3]
    for i in range(4):
        v1 = k[i] ^ rotl(k[i], 1) ^ rotl(k[i], 8)
        v2 = k[i] ^ rotl(k[i], 1) ^ rotl(k[i], 11)
        rk[i] = v1
        rk[(i + 4) ^ 1] = v2

    return rk

# -----------------------------
# Endpoints
# -----------------------------

def encrypt(data: bytes, key: bytes) -> bytes:
    rk = key_schedule(key)

    pad_len = 16 - (len(data) % 16)
    data += bytes([pad_len]) * pad_len

    out = bytearray(len(data))

    for off in range(0, len(data), 16):
        x0 = int.from_bytes(data[off:off+4], 'little')
        x1 = int.from_bytes(data[off+4:off+8], 'little')
        x2 = int.from_bytes(data[off+8:off+12], 'little')
        x3 = int.from_bytes(data[off+12:off+16], 'little')

        for i in range(80):
            rk_i = rk[i & 15]
            if i & 1:
                t = (rotl(x0, 8) ^ x2 ^ ((x1 + rk_i) & MASK)) & MASK
            else:
                t = (rotl(x0, 1) ^ x2 ^ ((x1 + rk_i) & MASK)) & MASK
            x0, x1, x2, x3 = x1, x2, x3, t

        out[off:off+4]   = x0.to_bytes(4, 'little')
        out[off+4:off+8] = x1.to_bytes(4, 'little')
        out[off+8:off+12]= x2.to_bytes(4, 'little')
        out[off+12:off+16]= x3.to_bytes(4, 'little')

    return bytes(out)


def decrypt(data: bytes, key: bytes) -> bytes:
    rk = key_schedule(key)
    out = bytearray(len(data))

    for off in range(0, len(data), 16):
        x0 = int.from_bytes(data[off:off+4], 'little')
        x1 = int.from_bytes(data[off+4:off+8], 'little')
        x2 = int.from_bytes(data[off+8:off+12], 'little')
        x3 = int.from_bytes(data[off+12:off+16], 'little')

        for i in range(79, -1, -1):
            rk_i = rk[i & 15]
            inner = (x3 ^ x1 ^ ((x0 + rk_i) & MASK)) & MASK
            if i & 1:
                old_x0 = rotr(inner, 8)
            else:
                old_x0 = rotr(inner, 1)
            x0, x1, x2, x3 = old_x0, x0, x1, x2

        out[off:off+4]   = x0.to_bytes(4, 'little')
        out[off+4:off+8] = x1.to_bytes(4, 'little')
        out[off+8:off+12]= x2.to_bytes(4, 'little')
        out[off+12:off+16]= x3.to_bytes(4, 'little')

    pad_len = out[-1]
    return bytes(out[:-pad_len])