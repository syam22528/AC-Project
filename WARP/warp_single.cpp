#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define ROUNDS 32

// ----------------------
// WARP 4-bit S-box
// ----------------------
static const uint8_t SBOX[16] = {
    0xc, 0x6, 0x9, 0x0,
    0x1, 0xa, 0x2, 0xb,
    0x3, 0x8, 0x5, 0xd,
    0x4, 0xe, 0x7, 0xf
};

// ----------------------
// Round constants
// ----------------------
static const uint8_t RC[ROUNDS] = {
    0x01,0x03,0x07,0x0f,0x1f,0x3e,0x3d,0x3b,
    0x37,0x2f,0x1e,0x3c,0x39,0x33,0x27,0x0e,
    0x1d,0x3a,0x35,0x2b,0x16,0x2c,0x18,0x30,
    0x21,0x02,0x05,0x0b,0x17,0x2e,0x1c,0x38
};

// ----------------------
// Subkey generation (toy version, for demo)
// ----------------------
void key_schedule(uint8_t master[10], uint8_t round_keys[ROUNDS][8]) {
    uint8_t k[10];
    memcpy(k, master, 10);

    for (int r = 0; r < ROUNDS; r++) {
        // simple rotation and S-box application
        for (int i = 0; i < 8; i++) {
            uint8_t hi = SBOX[k[i % 10] >> 4];
            uint8_t lo = SBOX[k[i % 10] & 0xf];
            round_keys[r][i] = (hi << 4) | lo;
        }

        // rotate master key for next round
        uint8_t tmp = k[0];
        for (int i = 0; i < 9; i++) k[i] = k[i+1];
        k[9] = tmp ^ RC[r]; // inject round constant
    }
}

// ----------------------
// 64-bit WARP encryption
// ----------------------
void warp_encrypt(uint8_t pt[8], uint8_t round_keys[ROUNDS][8], uint8_t ct[8]) {
    uint8_t state[8];
    memcpy(state, pt, 8);

    for (int r = 0; r < ROUNDS; r++) {
        // SubBytes
        for (int i = 0; i < 8; i++) {
            uint8_t hi = SBOX[state[i] >> 4];
            uint8_t lo = SBOX[state[i] & 0xf];
            state[i] = (hi << 4) | lo;
        }

        // AddRoundKey
        for (int i = 0; i < 8; i++) {
            state[i] ^= round_keys[r][i];
        }

        // Linear layer / permutation (simple rotation-based)
        uint8_t tmp[8];
        for (int i = 0; i < 8; i++) tmp[i] = (state[i] << 1) | (state[i] >> 7);
        memcpy(state, tmp, 8);
    }

    memcpy(ct, state, 8);
}

// ----------------------
// Benchmark / test
// ----------------------
int main() {
    uint8_t pt[8] = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77};
    uint8_t key[10] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09};
    uint8_t ct[8];
    uint8_t round_keys[ROUNDS][8];

    key_schedule(key, round_keys);

    int N = 1000000;
    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        warp_encrypt(pt, round_keys, ct);
    }

    clock_t end = clock();

    printf("Ciphertext: ");
    for (int i = 0; i < 8; i++) printf("%02x ", ct[i]);
    printf("\n");

    double t = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for %d encryptions: %f sec\n", N, t);

    return 0;
}