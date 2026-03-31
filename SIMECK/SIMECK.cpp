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

// ✅ SIMECK round function
inline uint32_t f(uint32_t x) {
    return (rotl(x, 5) & x) ^ rotl(x, 1);
}

// Encrypt one block
inline void simeck_encrypt(uint32_t &left, uint32_t &right, uint32_t round_keys[ROUNDS]) {

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

    uint32_t round_keys[ROUNDS];

    // Same simple keys as GPU/SIMON baseline
    for (int i = 0; i < ROUNDS; i++)
        round_keys[i] = i;

    cout << "===== SIMECK CPU OPENMP PERFORMANCE =====\n";

    for (int N : test_sizes) {

        vector<uint32_t> left(N), right(N);

        for (int i = 0; i < N; i++) {
            left[i] = i;
            right[i] = i ^ 0xdeadbeef;
        }

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            simeck_encrypt(left[i], right[i], round_keys);
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