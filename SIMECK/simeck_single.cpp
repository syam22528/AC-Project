#include <iostream>
#include <cstdint>
#include <chrono>
using namespace std;

#define ROUNDS 44

// ─────────────────────────────────────────────
//  Core operations
// ─────────────────────────────────────────────
inline uint32_t ROL32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t f(uint32_t x) {
    return (ROL32(x, 5) & x) ^ ROL32(x, 1);
}

// ─────────────────────────────────────────────
//  Key Schedule  —  Simeck-64/128
//
//  Key:   K = k3 || k2 || k1 || k0  (128 bits, 4 × 32-bit words)
//  Round key: rk[i] = k0
//  Update:    tmp  = f(k1) ^ k0 ^ C ^ z[i]
//             k0 ← k1 ← k2 ← k3 ← tmp
//
//  C  = 0xFFFFFFFC
//  z  = z_2 sequence (62-bit period, same as Simon-64/128)
// ─────────────────────────────────────────────
void keySchedule(const uint32_t key[4], uint32_t rk[ROUNDS]) {

    // z_2 sequence: 62-bit period, first 44 bits extracted here
    static const uint8_t z[44] = {
        1,1,1,1,1,0,1,0, 0,0,1,0,0,1,0,1,
        0,1,1,0,0,0,0,1, 1,1,0,0,1,1,0,1,
        1,1,1,1,0,1,0,0, 0,1,0,0
    };
    const uint32_t C = 0xFFFFFFFC;

    // key[0]=k0, key[1]=k1, key[2]=k2, key[3]=k3
    uint32_t k0 = key[0], k1 = key[1], k2 = key[2], k3 = key[3];

    for (int i = 0; i < ROUNDS; i++) {
        rk[i]        = k0;
        uint32_t tmp = f(k1) ^ k0 ^ C ^ (uint32_t)z[i];
        k0 = k1;
        k1 = k2;
        k2 = k3;
        k3 = tmp;
    }
}

// ─────────────────────────────────────────────
//  Encrypt one 64-bit block
// ─────────────────────────────────────────────
inline void encrypt(uint32_t &x, uint32_t &y, const uint32_t rk[ROUNDS]) {
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t tmp = x;
        x = y ^ f(x) ^ rk[i];
        y = tmp;
    }
}

// ─────────────────────────────────────────────
//  Decrypt one 64-bit block  (reverse Feistel)
// ─────────────────────────────────────────────
inline void decrypt(uint32_t &x, uint32_t &y, const uint32_t rk[ROUNDS]) {
    for (int i = ROUNDS - 1; i >= 0; i--) {
        uint32_t tmp = y;
        y = x ^ f(y) ^ rk[i];
        x = tmp;
    }
}

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
void printHex(const char *label, uint32_t a, uint32_t b) {
    printf("%-16s %08X %08X\n", label, a, b);
}

int main() {

    // ── 1. Key setup ──────────────────────────
    // Key (little-endian word order, matching Simon/Simeck convention):
    //   k0 = 0x03020100
    //   k1 = 0x0b0a0908
    //   k2 = 0x13121110
    //   k3 = 0x1b1a1918
    const uint32_t key[4] = { 0x03020100, 0x0b0a0908,
                               0x13121110, 0x1b1a1918 };
    uint32_t rk[ROUNDS];
    keySchedule(key, rk);

    // ── 2. Correctness: encrypt then decrypt ──
    printf("=== Simeck-64/128 Correctness Test ===\n");

    uint32_t pt_x = 0x65656877, pt_y = 0x6F6E6465; // "hewokend" in ASCII
    uint32_t x = pt_x, y = pt_y;

    printHex("Plaintext:", x, y);
    encrypt(x, y, rk);
    printHex("Ciphertext:", x, y);
    decrypt(x, y, rk);
    printHex("Decrypted:", x, y);
    printf("Roundtrip:  %s\n\n",
           (x == pt_x && y == pt_y) ? "PASS ✓" : "FAIL ✗");

    // ── 3. Zero key / zero plaintext vector ───
    printf("=== Zero-Key Test ===\n");
    const uint32_t zkey[4] = {0, 0, 0, 0};
    uint32_t zrk[ROUNDS];
    keySchedule(zkey, zrk);

    uint32_t zx = 0, zy = 0;
    printHex("Plaintext:", zx, zy);
    encrypt(zx, zy, zrk);
    printHex("Ciphertext:", zx, zy);
    decrypt(zx, zy, zrk);
    printHex("Decrypted:", zx, zy);
    printf("Roundtrip:  %s\n\n",
           (zx == 0 && zy == 0) ? "PASS ✓" : "FAIL ✗");

    // ── 4. Benchmark ──────────────────────────
    const int N = 1 << 20;
    uint32_t *bx = new uint32_t[N];
    uint32_t *by = new uint32_t[N];
    for (int i = 0; i < N; i++) { bx[i] = (uint32_t)i; by[i] = (uint32_t)(i + 1); }

    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) encrypt(bx[i], by[i], rk);
    auto t1 = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(t1 - t0).count();
    printf("=== Benchmark: %d blocks ===\n", N);
    printf("Time:       %.4f s\n",  elapsed);
    printf("Throughput: %.3f GB/s\n", (double)(N * 8) / elapsed / 1e9);

    delete[] bx;
    delete[] by;
    return 0;
}