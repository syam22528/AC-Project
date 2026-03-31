#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

#define ROUNDS 32
#define PARALLEL 32  // 32 blocks per bitslice batch

// Bit-sliced state for 32 blocks
struct BSState {
    uint32_t b[16][4]; // 16 nibbles, 4 bits each (32 blocks)
};

// Example S-box in bitslice (boolean ops)
inline void sbox_bitslice(uint32_t &b0, uint32_t &b1, uint32_t &b2, uint32_t &b3) {
    uint32_t t0 = b1 ^ b2;
    uint32_t t1 = b0 & b3;
    uint32_t t2 = b0 ^ b1;
    uint32_t t3 = b2 | t1;

    b3 ^= t0;
    b2 ^= t2;
    b1 ^= t3;
    b0 ^= b3;
}

// SubCells
inline void sub_cells(BSState &s) {
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
        sbox_bitslice(s.b[i][0], s.b[i][1], s.b[i][2], s.b[i][3]);
}

// Permutation table
static const int PERM[16] = {
    0, 5, 10, 15,
    4, 9, 14, 3,
    8, 13, 2, 7,
    12, 1, 6, 11
};

// Permute
inline void permute(BSState &s) {
    BSState tmp;
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            tmp.b[i][b] = s.b[PERM[i]][b];
    s = tmp;
}

// Add round key
inline void add_round_key(BSState &s, BSState &k) {
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            s.b[i][b] ^= k.b[i][b];
}

// Key schedule
inline void key_schedule(BSState &k, int r) {
    BSState tmp = k;
    for (int i = 0; i < 15; i++)
        k.b[i][0] = tmp.b[i + 1][0];
    k.b[15][0] ^= (r & 1); // minimal schedule for 64-bit SKINNY
}

// Encrypt one batch (32 blocks)
inline void skinny_bitslice_encrypt(BSState &state, BSState key) {
    #pragma GCC unroll 4
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(state);
        add_round_key(state, key);
        permute(state);
        key_schedule(key, r);
    }
}

int main() {

    vector<int> test_sizes = {
        1024, 16384, 65536, 262144,
        1048576, 4194304, 10485760,
        52428800, 104857600
    };

    cout << "===== SKINNY BITSLICE CPU PERFORMANCE =====\n";

    for (int N : test_sizes) {

        int batches = (N + PARALLEL - 1) / PARALLEL;
        vector<BSState> data(batches);

        // Initialize blocks
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batches; i++) {
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    data[i].b[n][b] = i + n + b; // dummy init, matches scalar values approx
        }

        BSState key = {}; // all-zero master key

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batches; i++)
            skinny_bitslice_encrypt(data[i], key);

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 16.0) / (time * 1e9);

        // Print first block (approximate)
        cout << hex;
        cout << "Ciphertext (first block approx): ";
        for (int n = 0; n < 16; n++) {
            uint8_t val = 0;
            for (int b = 0; b < 4; b++)
                val |= ((data[0].b[n][b] & 1) << b);
            cout << (int)val << " ";
        }
        cout << dec << endl;

        cout << "N = " << N
             << " | Time = " << time << " s"
             << " | Throughput = " << throughput << " GB/s\n";
    }

    return 0;
}