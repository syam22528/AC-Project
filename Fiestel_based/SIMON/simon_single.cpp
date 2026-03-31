#include <iostream>
#include <cstdint>
#include <chrono>

using namespace std;

#define ROUNDS 44

// Left rotation
inline uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// Right rotation
inline uint32_t rotr32(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
}

// Non-linear round function
inline uint32_t f(uint32_t x) {
    return (rotl32(x, 1) & rotl32(x, 8)) ^ rotl32(x, 2);
}

// Core Encryption
void simon_encrypt(uint32_t &l, uint32_t &r, const uint32_t round_keys[ROUNDS]) {
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t temp = l;
        l = r ^ f(l) ^ round_keys[i];
        r = temp;
    }
}

// SIMON 64/128 Key Schedule
void simon_key_schedule(const uint32_t master_key[4], uint32_t round_keys[ROUNDS]) {
    const uint64_t z3 = 0b11110000101100111001010001001000000111101001100011010111011011ULL;
    const uint32_t c = 0xfffffffc;

    // Copy master key as first 4 round keys
    for (int i = 0; i < 4; i++)
        round_keys[i] = master_key[i];

    for (int i = 4; i < ROUNDS; i++) {
        uint32_t tmp = rotr32(round_keys[i - 1], 3);
        tmp ^= round_keys[i - 3];
        tmp ^= rotr32(tmp, 1);

        // Get i-th bit of z3 (mod 62)
        uint32_t z_bit = (z3 >> ((i - 4) % 62)) & 1;

        round_keys[i] = c ^ z_bit ^ tmp ^ round_keys[i - 4];
    }
}

int main() {
    uint32_t left = 0x656b696c;
    uint32_t right = 0x20646e75;

    uint32_t master_key[4] = {0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918};
    uint32_t round_keys[ROUNDS];

    simon_key_schedule(master_key, round_keys);

    const int N = 4194304; 
    uint32_t l = left;
    uint32_t r = right;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        simon_encrypt(l, r, round_keys);
    }
    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double>(end - start).count();
    double throughput = (N * 8.0) / (time * 1e9);

    cout << "Output: " << hex << l << " " << r << endl;
    cout << dec;
    cout << "Time: " << time << " s\n";
    cout << "Throughput: " << throughput << " GB/s\n";

    return 0;
}