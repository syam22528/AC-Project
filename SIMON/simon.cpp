#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std;

#define ROUNDS 44

// Z3 sequence
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
    return (rotl(x, 1) & rotl(x, 8)) ^ rotl(x, 2);
}

// Key expansion
void key_expansion(uint32_t key[4], uint32_t round_keys[ROUNDS]) {
    const uint32_t c = 0xfffffffc;

    for (int i = 0; i < 4; i++)
        round_keys[i] = key[i];

    for (int i = 4; i < ROUNDS; i++) {
        uint32_t tmp = rotr(round_keys[i - 1], 3);
        tmp ^= round_keys[i - 3];
        tmp ^= rotr(tmp, 1);

        round_keys[i] = (~round_keys[i - 4]) ^ tmp ^ Z3[(i - 4) % 62] ^ c;
    }
}

// Encrypt one block
inline void simon_encrypt(uint32_t &left, uint32_t &right, uint32_t round_keys[ROUNDS]) {

    #pragma GCC unroll 4
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t tmp = left;
        left = right ^ f(left) ^ round_keys[i];
        right = tmp;
    }
}

int main() {

    vector<int> test_sizes = {
        1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800 , 104857600
    };

    uint32_t key[4] = {1, 2, 3, 4};
    uint32_t round_keys[ROUNDS];

    key_expansion(key, round_keys);

    cout << "===== CPU OPENMP PERFORMANCE =====\n";

    for (int N : test_sizes) {

        vector<uint32_t> left(N), right(N);

        for (int i = 0; i < N; i++) {
            left[i] = i;
            right[i] = i ^ 0xdeadbeef;
        }

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            simon_encrypt(left[i], right[i], round_keys);
        }

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 8.0) / (time * 1e9);

        // ✅ PRINT CIPHERTEXT (FIRST BLOCK ONLY)
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