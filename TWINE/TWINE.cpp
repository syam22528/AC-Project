#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstring>

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
    uint32_t y0 = (~x0 & ~x1 & ~x2 & ~x3) | (~x0 & x1 & x2 & ~x3) | (x0 & ~x1 & x2 & x3) | (x0 & x1 & ~x2 & x3);
    uint32_t y1 = (~x0 & x1 & ~x2 & ~x3) | (x0 & ~x1 & ~x2 & x3) | (x0 & x1 & x2 & ~x3) | (~x0 & ~x1 & x2 & x3);
    uint32_t y2 = (~x0 & ~x1 & x2 & ~x3) | (x0 & ~x1 & ~x2 & ~x3) | (~x0 & x1 & x2 & x3) | (x0 & x1 & ~x2 & x3);
    uint32_t y3 = (~x0 & ~x1 & ~x2 & x3) | (~x0 & x1 & x2 & x3) | (x0 & ~x1 & x2 & ~x3) | (x0 & x1 & ~x2 & ~x3);

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
// Round keys storage (bit-sliced)
// ---------------------------
uint32_t round_keys[ROUNDS][16][4];

// ---------------------------
// S-box for key schedule
// ---------------------------
uint8_t SBOX[16] = {0xC,0x0,0xF,0xA,0x2,0xB,0x9,0x5,0x8,0x3,0xD,0x7,0x1,0xE,0x6,0x4};

// ---------------------------
// Key schedule (80-bit key → 36 × 16 × 4 bits)
// ---------------------------
void key_schedule(uint8_t key[20]) {
    uint8_t k[20];
    memcpy(k, key, 20);

    for (int r = 0; r < ROUNDS; r++) {
        for (int n = 0; n < 16; n++)
            for (int b = 0; b < 4; b++)
                round_keys[r][n][b] = (k[n] >> b) & 1u;

        // Rotate key nibbles
        uint8_t temp[20];
        for (int i = 0; i < 20; i++)
            temp[i] = k[(i + 13) % 20];
        memcpy(k, temp, 20);

        // S-box on first 4 nibbles
        for (int i = 0; i < 4; i++)
            k[i] = SBOX[k[i]];

        // XOR round counter to last nibble
        k[19] ^= r & 0xF;
    }
}

// ---------------------------
// Encrypt batch
// ---------------------------
inline void twine_bitslice_encrypt(BSState &state) {
    for (int r = 0; r < ROUNDS; r++) {
        for (int n = 0; n < 16; n++)
            for (int b = 0; b < 4; b++)
                state.b[n][b] ^= round_keys[r][n][b];

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

    // 80-bit master key (same as CUDA version)
    uint8_t master_key[20] = {
        0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,
        0xA,0xB,0xC,0xD,0xE,0xF,0x0,0x1,0x2,0x3
    };

    key_schedule(master_key);

    cout << "===== TWINE BITSLICE CPU PERFORMANCE (aligned plaintext) =====\n";

    for (int N : test_sizes) {
        int batches = (N + PARALLEL - 1) / PARALLEL;
        vector<BSState> data(batches);

        // Initialize plaintext → bitslice
        #pragma omp parallel for
        for (int i = 0; i < batches; i++) {
            memset(&data[i], 0, sizeof(BSState));

            for (int blk = 0; blk < PARALLEL; blk++) {
                int global_idx = i * PARALLEL + blk;
                if (global_idx >= N) break;

                for (int n = 0; n < 16; n++) {
                    uint8_t val = global_idx % 16; // matches CUDA pattern
                    for (int b = 0; b < 4; b++)
                        if (val & (1 << b))
                            data[i].b[n][b] |= (1u << blk);
                }
            }
        }

        // Encrypt
        auto start = chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < batches; i++)
            twine_bitslice_encrypt(data[i]);
        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 16.0) / (time * 1e9);

        // First block ciphertext
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