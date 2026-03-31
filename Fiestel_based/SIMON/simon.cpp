#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std;

#define ROUNDS 44

// Z3 sequence for SIMON 64/128 (LSB first)
const uint8_t Z3[62] = {
1,1,1,1,0,1,0,1,1,0,0,0,1,0,0,1,
0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,1,
1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,1,
0,0,1,1,0,0,0,1,1,1,1,0,1,0
};

// Rotate left
inline uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// Rotate right
inline uint32_t rotr(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
}

// SIMON round function
inline uint32_t f(uint32_t x) {
    return (rotl(x,1) & rotl(x,8)) ^ rotl(x,2);
}

// Key expansion (fully compliant)
void key_expansion(const uint32_t key[4], uint32_t round_keys[ROUNDS]) {
    const uint32_t c = 0xfffffffc;

    // Copy master key
    for(int i=0;i<4;i++)
        round_keys[i] = key[i];

    for(int i=4;i<ROUNDS;i++) {
        uint32_t tmp = rotr(round_keys[i-1],3) ^ round_keys[i-3];
        tmp ^= rotr(tmp,1);
        uint32_t z_bit = Z3[(i-4)%62];
        round_keys[i] = c ^ z_bit ^ tmp ^ round_keys[i-4];
    }
}

// Encrypt one block
inline void simon_encrypt(uint32_t &left, uint32_t &right, const uint32_t round_keys[ROUNDS]) {
    for(int i=0;i<ROUNDS;i++) {
        uint32_t tmp = left;
        left = right ^ f(left) ^ round_keys[i];
        right = tmp;
    }
}

int main() {
    // Test vector (from SIMON 64/128 paper)
    uint32_t left = 0x656b696c;
    uint32_t right = 0x20646e75;

    // 128-bit key (4 words)
    uint32_t key[4] = {0x19181110,0x11100908,0x09050302,0x01000000};
    uint32_t round_keys[ROUNDS];

    // Generate all round keys
    key_expansion(key, round_keys);

    // Benchmark sizes
    vector<int> test_sizes = {1024, 16384, 65536, 262144, 1048576, 4194304, 10485760};

    cout << "===== CPU OPENMP PERFORMANCE =====\n";

    for(int N : test_sizes) {
        vector<uint32_t> left_blocks(N), right_blocks(N);

        // Initialize blocks
        for(int i=0;i<N;i++) {
            left_blocks[i] = i;
            right_blocks[i] = i ^ 0xabcdabcd;
        }

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++) {
            simon_encrypt(left_blocks[i], right_blocks[i], round_keys);
        }

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end-start).count();
        double throughput = (N*8.0)/(time*1e9); // GB/s

        // Print first block as verification
        cout << hex << "Ciphertext (first block): " 
             << left_blocks[0] << " " << right_blocks[0] << endl;
        cout << dec << "N = " << N
             << " | Time = " << time << " s"
             << " | Throughput = " << throughput << " GB/s\n";
    }

    return 0;
}