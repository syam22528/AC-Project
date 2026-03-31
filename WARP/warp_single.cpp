#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define ROUNDS 32

// Simple 4-bit S-box (like lightweight designs)
static const uint8_t SBOX[16] = {
    0xC,0x6,0x9,0x0,0x1,0xA,0x2,0xB,
    0x3,0x8,0x5,0xD,0x4,0xE,0x7,0xF
};

// Substitute each nibble
static inline void sub_bytes(uint8_t state[16]) {
    for (int i = 0; i < 16; i++) {
        uint8_t hi = SBOX[state[i] >> 4];
        uint8_t lo = SBOX[state[i] & 0xF];
        state[i] = (hi << 4) | lo;
    }
}

// Simple permutation (GFN-style shuffle)
static inline void permute(uint8_t state[16]) {
    uint8_t tmp[16];
    for (int i = 0; i < 16; i++) {
        tmp[i] = state[(i * 5) % 16];
    }
    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

// Round function
static inline void round_func(uint8_t state[16], uint8_t key[16], int r) {
    sub_bytes(state);

    for (int i = 0; i < 16; i++) {
        state[i] ^= key[i] ^ r;
    }

    permute(state);
}

void warp_encrypt(uint8_t pt[16], uint8_t key[16], uint8_t ct[16]) {
    uint8_t state[16];

    for (int i = 0; i < 16; i++)
        state[i] = pt[i];

    for (int r = 0; r < ROUNDS; r++) {
        round_func(state, key, r);
    }

    for (int i = 0; i < 16; i++)
        ct[i] = state[i];
}

// Benchmark
int main() {
    uint8_t pt[16] = {0};
    uint8_t key[16] = {0};
    uint8_t ct[16];

    int N = 1000000;

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        warp_encrypt(pt, key, ct);
    }

    clock_t end = clock();

    double t = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Ciphertext: ");
    for (int i = 0; i < 16; i++) printf("%02x ", ct[i]);
    printf("\nTime: %f sec\n", t);

    return 0;
}