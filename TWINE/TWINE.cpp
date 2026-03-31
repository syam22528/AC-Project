#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

#define ROUNDS 36
#define PARALLEL 32  // 32 blocks per batch

// ---------------------------
// Bit-sliced state
// ---------------------------
struct BSState {
    uint32_t b[16][4]; // 16 nibbles × 4 bits
};

// ---------------------------
// TWINE S-box (bitsliced)
// ---------------------------
inline void sbox_bitslice(uint32_t &x0, uint32_t &x1,
                          uint32_t &x2, uint32_t &x3) {

    // Temporary variables (derived manually / generic logic)
    uint32_t y0 = (~x0 & ~x1 & ~x2 & ~x3) |
                  (~x0 &  x1 &  x2 & ~x3) |
                  ( x0 & ~x1 &  x2 &  x3) |
                  ( x0 &  x1 & ~x2 &  x3);

    uint32_t y1 = (~x0 &  x1 & ~x2 & ~x3) |
                  ( x0 & ~x1 & ~x2 &  x3) |
                  ( x0 &  x1 &  x2 & ~x3) |
                  (~x0 & ~x1 &  x2 &  x3);

    uint32_t y2 = (~x0 & ~x1 &  x2 & ~x3) |
                  ( x0 & ~x1 & ~x2 & ~x3) |
                  (~x0 &  x1 &  x2 &  x3) |
                  ( x0 &  x1 & ~x2 &  x3);

    uint32_t y3 = (~x0 & ~x1 & ~x2 &  x3) |
                  (~x0 &  x1 &  x2 &  x3) |
                  ( x0 & ~x1 &  x2 & ~x3) |
                  ( x0 &  x1 & ~x2 & ~x3);

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
// TWINE Permutation
// ---------------------------
static const int PERM[16] = {
    0,9,2,13,4,11,6,15,
    8,1,10,3,12,5,14,7
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
// Encrypt batch (32 blocks)
// ---------------------------
inline void twine_bitslice_encrypt(BSState &state) {

    #pragma GCC unroll 4
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(state);
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

    cout << "===== TWINE BITSLICE CPU PERFORMANCE =====\n";

    for (int N : test_sizes) {

        int batches = (N + PARALLEL - 1) / PARALLEL;
        vector<BSState> data(batches);

        // ---------------------------
        // Initialize (convert scalar → bitslice)
        // ---------------------------
        #pragma omp parallel for
        for (int i = 0; i < batches; i++) {
            for (int n = 0; n < 16; n++) {
                for (int b = 0; b < 4; b++) {
                    data[i].b[n][b] = 0;
                }
            }

            for (int blk = 0; blk < PARALLEL; blk++) {
                int global_idx = i * PARALLEL + blk;

                if (global_idx >= N) break;

                for (int n = 0; n < 16; n++) {
                    uint8_t val = (global_idx * 16 + n) % 16;

                    for (int b = 0; b < 4; b++) {
                        if (val & (1 << b))
                            data[i].b[n][b] |= (1u << blk);
                    }
                }
            }
        }

        // ---------------------------
        // Encrypt
        // ---------------------------
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < batches; i++)
            twine_bitslice_encrypt(data[i]);

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 16.0) / (time * 1e9);

        // ---------------------------
        // Convert back (first block)
        // ---------------------------
        cout << hex;
        cout << "Ciphertext (first block): ";

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