#include <iostream>
#include <cstdint>
#include <cstring>
#include <chrono>
using namespace std;

#define ROUNDS 36
#define N (1 << 20)  // Number of blocks to encrypt

// ---------------------
// 4-bit S-box
// ---------------------
static const uint8_t SBOX[16] = {
    0xC,0x0,0xF,0xA,0x2,0xB,0x9,0x5,
    0x8,0x3,0xD,0x7,0x1,0xE,0x6,0x4
};

// ---------------------
// Permutation layer
// ---------------------
static const uint8_t P[16] = {
    0,9,2,13,4,11,6,15,8,1,10,3,12,5,14,7
};

// ---------------------
// Round keys storage
// ---------------------
uint8_t round_keys[ROUNDS][16];

// ---------------------
// Key schedule for TWINE-80
// key: 80-bit key stored as 20 nibbles
// ---------------------
void key_schedule(uint8_t key[20]) {
    uint8_t k[20];
    memcpy(k, key, 20);

    for (int r = 0; r < ROUNDS; r++) {
        // Extract round key: take first 16 nibbles
        memcpy(round_keys[r], k, 16);

        // Rotate key nibbles for next round
        uint8_t temp[20];
        for (int i = 0; i < 20; i++)
            temp[i] = k[(i + 13) % 20];
        memcpy(k, temp, 20);

        // Apply S-box to first 4 nibbles
        for (int i = 0; i < 4; i++)
            k[i] = SBOX[k[i]];

        // XOR round counter to last nibble
        k[19] ^= r & 0xF;
    }
}

// ---------------------
// Encrypt 64-bit block (16 nibbles)
// ---------------------
void encrypt(uint8_t state[16]) {
    uint8_t tmp[16];

    for (int r = 0; r < ROUNDS; r++) {
        // Add round key
        for (int i = 0; i < 16; i++)
            state[i] ^= round_keys[r][i];

        // S-box
        for (int i = 0; i < 16; i++)
            state[i] = SBOX[state[i]];

        // Permutation
        for (int i = 0; i < 16; i++)
            tmp[i] = state[P[i]];

        memcpy(state, tmp, 16);
    }
}

// ---------------------
// Main: benchmarking
// ---------------------
int main() {
    // Example 80-bit key
    uint8_t key[20] = {
        0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,
        0xA,0xB,0xC,0xD,0xE,0xF,0x0,0x1,0x2,0x3
    };

    key_schedule(key);  // Generate round keys

    // Allocate N plaintext blocks (each 16 nibbles)
    uint8_t *data = new uint8_t[N * 16];
    for (int i = 0; i < N * 16; i++)
        data[i] = i % 16;  // simple test pattern

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++)
        encrypt(&data[i * 16]);

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();
    double throughput = (double)(N * 16) / time / 1e9; // GB/s

    cout << "Time: " << time << " s\n";
    cout << "Throughput: " << throughput << " GB/s\n";

    delete[] data;
}