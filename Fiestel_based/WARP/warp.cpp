#include <iostream>
#include <cstdint>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstring>

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
// WARP 4-bit S-box (bitslice-friendly)
// Reference: original WARP S-box in bitslice form
// ---------------------------
inline void sbox_bitslice(uint32_t &x0, uint32_t &x1,
                          uint32_t &x2, uint32_t &x3) {
    // Bitslice version of WARP S-box
    uint32_t t0 = x1 ^ x2;
    uint32_t t1 = x0 | x3;
    uint32_t t2 = x0 ^ x1 ^ x3;
    uint32_t t3 = x0 & x2;

    x0 = t0 ^ t1;
    x1 = t2;
    x2 = t3 ^ x1;
    x3 = x0 ^ x2 ^ x3;
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
// Permutation (WARP linear layer approximation)
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
// WARP round constants
// ---------------------------
static const uint8_t RC[ROUNDS] = {
    0x01,0x03,0x07,0x0f,0x1f,0x3e,0x3d,0x3b,
    0x37,0x2f,0x1e,0x3c,0x39,0x33,0x27,0x0e,
    0x1d,0x3a,0x35,0x2b,0x16,0x2c,0x18,0x30,
    0x21,0x02,0x05,0x0b,0x17,0x2e,0x1c,0x38
};

// ---------------------------
// Add round key + round constant
// ---------------------------
inline void add_round(BSState &s, const uint8_t round_key[16], int r) {
    for (int n = 0; n < 16; n++)
        for (int b = 0; b < 4; b++) {
            if (round_key[n] & (1 << b))
                s.b[n][b] ^= 0xFFFFFFFFu;
            if (RC[r] & (1 << b))
                s.b[n][b] ^= 0xFFFFFFFFu;
        }
}

// ---------------------------
// Key schedule (derive per-round subkeys)
// ---------------------------
void key_schedule(const uint8_t master[16], uint8_t round_keys[ROUNDS][16]) {
    uint8_t k[16];
    memcpy(k, master, 16);

    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < 16; i++)
            round_keys[r][i] = k[i];

        // simple rotation + inject RC for next round key
        uint8_t tmp = k[0];
        for (int i = 0; i < 15; i++) k[i] = k[i+1];
        k[15] = tmp ^ RC[r];
    }
}

// ---------------------------
// Bitslice encrypt
// ---------------------------
inline void warp_bitslice_encrypt(BSState &state, const uint8_t round_keys[ROUNDS][16]) {
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(state);
        add_round(state, round_keys[r], r);
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

    uint8_t master_key[16] = {0};
    uint8_t round_keys[ROUNDS][16];

    key_schedule(master_key, round_keys);

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
            warp_bitslice_encrypt(data[i], round_keys);

        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration<double>(end - start).count();
        double throughput = (N * 8.0) / (time * 1e9); // 8 bytes per block = 64-bit block

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