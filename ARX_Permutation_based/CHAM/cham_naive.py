MASK = 0xFFFFFFFF

def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & MASK

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK

def key_schedule(key: bytes) -> list:
    assert len(key) == 16
    k = [int.from_bytes(key[i:i+4], 'little') for i in range(0, 16, 4)]
    rk = [0] * 16
    for i in range(4):
        rk[i]       = k[i] ^ rotl(k[i], 1) ^ rotl(k[i], 8)
        rk[(i+4)^1] = k[i] ^ rotl(k[i], 1) ^ rotl(k[i], 11)
    return rk

def _enc_block(block: bytes, rk: list) -> bytes:
    x = [int.from_bytes(block[i:i+4], 'little') for i in range(0, 16, 4)]
    for i in range(80):
        if i % 2 == 0:
            t = (rotl(x[0], 1) ^ x[2] ^ ((x[1] + rk[i % 16]) & MASK)) & MASK
        else:
            t = (rotl(x[0], 8) ^ x[2] ^ ((x[1] + rk[i % 16]) & MASK)) & MASK
        x[0], x[1], x[2], x[3] = x[1], x[2], x[3], t
    return b''.join(w.to_bytes(4, 'little') for w in x)

def _dec_block(block: bytes, rk: list) -> bytes:
    x = [int.from_bytes(block[i:i+4], 'little') for i in range(0, 16, 4)]
    for i in range(79, -1, -1):
        inner = (x[3] ^ x[1] ^ ((x[0] + rk[i % 16]) & MASK)) & MASK
        if i % 2 == 0:
            old_x0 = rotr(inner, 1)
        else:
            old_x0 = rotr(inner, 8)
        x[0], x[1], x[2], x[3] = old_x0, x[0], x[1], x[2]
    return b''.join(w.to_bytes(4, 'little') for w in x)

def _pad(data: bytes) -> bytes:
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)

def _unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    assert 1 <= pad_len <= 16, "Invalid padding"
    assert data[-pad_len:] == bytes([pad_len] * pad_len), "Corrupt padding"
    return data[:-pad_len]

# -----------------------------
# Endpoints
# -----------------------------
def encrypt(data: bytes, key: bytes) -> bytes:
    rk = key_schedule(key)
    padded = _pad(data)
    return b''.join(_enc_block(padded[i:i+16], rk) for i in range(0, len(padded), 16))

def decrypt(data: bytes, key: bytes) -> bytes:
    assert len(data) % 16 == 0
    rk = key_schedule(key)
    decrypted = b''.join(_dec_block(data[i:i+16], rk) for i in range(0, len(data), 16))
    return _unpad(decrypted)