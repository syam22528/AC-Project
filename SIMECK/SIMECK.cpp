#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std;

#define ROUNDS 44

// Rotate left
inline uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// SIMECK round function
inline uint32_t f(uint32_t x) {
    return (rotl(x, 5) & x) ^ rotl(x, 1);
}

// SIMECK Key Schedule (Expands 128-bit master key into 44 32-bit round keys)
inline void simeck_key_schedule(const uint32_t master_key[4], uint32_t round_keys[ROUNDS]) {
    uint32_t C = 0xFFFFFFFC; // Constant C = 2^32 - 4
    
    // Initial state from master key
    uint32_t k  = master_key[0];
    uint32_t t0 = master_key[1];
    uint32_t t1 = master_key[2];
    uint32_t t2 = master_key[3];
    
    // Z1 sequence LFSR (initial state: 6 bits of 1)
    uint32_t lfsr = 0x3F; 

    for (int i = 0; i < ROUNDS; i++) {
        round_keys[i] = k; // Store the current round key
        
        // Get the lowest bit of the LFSR for the sequence Z1
        uint32_t z_i = lfsr & 1;
        
        // LFSR update using primitive polynomial X^6 + X + 1
        uint32_t new_bit = (lfsr ^ (lfsr >> 1)) & 1;
        lfsr = (lfsr >> 1) | (new_bit << 5);

        // Calculate the next t value
        uint32_t tmp = k ^ f(t0) ^ C ^ z_i;
        
        // Shift registers
        k  = t0;
        t0 = t1;
        t1 = t2;
        t2 = tmp;
    }
}

// Encrypt one block
inline void simeck_encrypt(uint32_t &left, uint32_t &right, const uint32_t round_keys[ROUNDS]) {
    #pragma GCC unroll 4
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t tmp = left;
        left = right ^ f(left) ^ round_keys[i];
        right = tmp;
    }
}

int main() {
    vector<int> test_sizes = {
        1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600
    };

    // 128-bit Master Key (4 x 32-bit words)
    uint32_t master_key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
    uint32_t round_keys[ROUNDS];

    // Pre-compute the key schedule once before the parallel loop
    simeck_key_schedule(master_key, round_keys);

    cout << "===== SIMECK64/128 CPU OPENMP PERFORMANCE =====\n";

    for (int N : test_sizes) {
        vector<uint32_t> left(N), right(N);

        for (int i = 0; i < N; i++) {
            left[i] = i;
            right[i] = i ^ 0xabcdabcd;
        }

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            simeck_encrypt(left[i], right[i], round_keys);
        }

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 8.0) / (time * 1e9);

        cout << hex;
        cout << "Ciphertext (first block): "
             << left[0] << " " << right[0] << endl;
        cout << dec;

        cout << "N = " << N
             << " | Time = " << time << " s"
             << " | Throughput = " << throughput << " GB/s\n";
    }

    return 0;
}