#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

#define ROUNDS 32
#define PARALLEL 32  // number of blocks per batch

// ---------------------------
// Bitsliced state for warp_encrypt
// ---------------------------
struct BSState {
    uint32_t b[16][4]; // 16 nibbles × 4 bits
};

// ---------------------------
// Original 4-bit S-box (bitslice-friendly)
// ---------------------------
inline void sbox_bitslice(uint32_t &x0, uint32_t &x1,
                          uint32_t &x2, uint32_t &x3) {
    // A simple nonlinear mixing similar to warp_encrypt
    uint32_t y0 = (~x0 & x1) ^ x2;
    uint32_t y1 = (x0 | x1) ^ x3;
    uint32_t y2 = (x0 & ~x2) ^ x3;
    uint32_t y3 = x0 ^ x1 ^ x2 ^ x3;

    x0 = y0;
    x1 = y1;
    x2 = y2;
    x3 = y3;
}

// ---------------------------
// SubCells
// ---------------------------
inline void sub_cells(BSState &s) {
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
        sbox_bitslice(s.b[i][0], s.b[i][1], s.b[i][2], s.b[i][3]);
}

// ---------------------------
// Permutation (like warp_encrypt)
// ---------------------------
static const int PERM[16] = {
    0,5,10,15,4,9,14,3,
    8,13,2,7,12,1,6,11
};

inline void permute(BSState &s) {
    BSState tmp;
    #pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            tmp.b[i][b] = s.b[PERM[i]][b];
    s = tmp;
}

// ---------------------------
// XOR with key + round constant (bitslice style)
// ---------------------------
inline void add_round(BSState &s, const uint8_t key[16], int r) {
    for (int n = 0; n < 16; n++) {
        for (int b = 0; b < 4; b++) {
            if (key[n] & (1 << b))
                s.b[n][b] ^= 0xFFFFFFFFu;  // all blocks
            if (r & (1 << b))
                s.b[n][b] ^= 0xFFFFFFFFu;
        }
    }
}

// ---------------------------
// Encrypt single bitsliced block
// ---------------------------
inline void warp_bitslice_encrypt(BSState &state, const uint8_t key[16]) {
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(state);
        add_round(state, key, r);
        permute(state);
    }
}

// ---------------------------
// MAIN
// ---------------------------
int main() {
    vector<int> test_sizes = {
        1024, 16384, 65536, 262144,
        1048576, 4194304, 10485760,
        52428800, 104857600
    };

    uint8_t key[16] = {0};

    cout << "===== WARP BITSLICE + OpenMP PERFORMANCE =====\n";

    for (int N : test_sizes) {
        int batches = (N + PARALLEL - 1) / PARALLEL;
        vector<BSState> data(batches);

        // ---------------------------
        // Initialize scalar → bitslice
        // ---------------------------
        #pragma omp parallel for
        for (int i = 0; i < batches; i++) {
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    data[i].b[n][b] = 0;

            for (int blk = 0; blk < PARALLEL; blk++) {
                int global_idx = i * PARALLEL + blk;
                if (global_idx >= N) break;

                for (int n = 0; n < 16; n++) {
                    uint8_t val = (global_idx * 16 + n) % 16;
                    for (int b = 0; b < 4; b++)
                        if (val & (1 << b))
                            data[i].b[n][b] |= (1u << blk);
                }
            }
        }

        // ---------------------------
        // Encrypt
        // ---------------------------
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < batches; i++)
            warp_bitslice_encrypt(data[i], key);

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 16.0) / (time * 1e9); // GB/s

        // ---------------------------
        // Convert first block back
        // ---------------------------
        cout << hex << "Ciphertext (first block): ";
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