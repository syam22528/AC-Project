#include <iostream>
#include <vector>
#include <cuda.h>
#include <cstdint>

#define ROUNDS 36
#define PARALLEL 32   // 32 blocks per bitslice batch

// ---------------------------
// Bit-sliced state
// ---------------------------
struct BSState {
    uint32_t b[16][4]; // 16 nibbles × 4 bits (32 blocks)
};

// ---------------------------
// TWINE S-box (bitsliced)
// ---------------------------
__device__ inline void sbox_bitslice(uint32_t &x0, uint32_t &x1,
                                     uint32_t &x2, uint32_t &x3) {
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

__device__ inline void sub_cells(BSState &s) {
    #pragma unroll
    for (int i = 0; i < 16; i++)
        sbox_bitslice(s.b[i][0], s.b[i][1], s.b[i][2], s.b[i][3]);
}

// ---------------------------
// TWINE Permutation table
// ---------------------------
__device__ const int PERM[16] = {
    0, 9, 2,13,
    4,11, 6,15,
    8, 1,10, 3,
   12, 5,14, 7
};

__device__ inline void permute(BSState &s) {
    BSState tmp;
    #pragma unroll
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            tmp.b[i][b] = s.b[PERM[i]][b];
    s = tmp;
}

// ---------------------------
// GPU kernel: TWINE encrypt batch
// ---------------------------
__global__ void twine_kernel(BSState *d_data, int batches, uint32_t rk[ROUNDS][16][4]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < batches; i += stride) {
        BSState s = d_data[i];

        for (int r = 0; r < ROUNDS; r++) {
            // Add round key
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    s.b[n][b] ^= rk[r][n][b];

            sub_cells(s);
            permute(s);
        }

        d_data[i] = s;
    }
}

// ---------------------------
// Host round keys (bit-sliced)
// ---------------------------
uint32_t round_keys[ROUNDS][16][4];

// TWINE S-box for CPU key schedule
uint8_t SBOX[16] = {
    0xC,0x0,0xF,0xA,0x2,0xB,0x9,0x5,
    0x8,0x3,0xD,0x7,0x1,0xE,0x6,0x4
};

// ---------------------------
// Key schedule (80-bit master key → 36×16×4 bits)
// ---------------------------
void key_schedule(uint8_t key[20]) {
    uint8_t k[20];
    memcpy(k, key, 20);

    for (int r = 0; r < ROUNDS; r++) {
        // First 16 nibbles → round key
        for (int n = 0; n < 16; n++)
            for (int b = 0; b < 4; b++)
                round_keys[r][n][b] = (k[n] >> b) & 1u;

        // Rotate key nibbles
        uint8_t temp[20];
        for (int i = 0; i < 20; i++)
            temp[i] = k[(i + 13) % 20];
        memcpy(k, temp, 20);

        // Apply S-box to first 4 nibbles
        for (int i = 0; i < 4; i++)
            k[i] = SBOX[k[i]];

        // XOR round counter to last nibble
        k[19] ^= r & 0xF;
    }
}

// ---------------------------
// MAIN
// ---------------------------
int main() {
    int test_sizes[] = {
        1024, 16384, 65536, 262144,
        1048576, 4194304, 10485760,
        52428800, 104857600
    };

    // Example 80-bit master key (20 nibbles)
    uint8_t master_key[20] = {
        0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,
        0xA,0xB,0xC,0xD,0xE,0xF,0x0,0x1,0x2,0x3
    };

    key_schedule(master_key);

    std::cout << "===== GPU BIT-SLICE TWINE PERFORMANCE =====\n";

    for (int t = 0; t < sizeof(test_sizes)/sizeof(int); t++) {
        int N = test_sizes[t];
        int batches = (N + PARALLEL - 1) / PARALLEL;

        // ---------------------------
        // Initialize plaintext bitslice (match CPU)
        // ---------------------------
        std::vector<BSState> h_data(batches);
        for (int i = 0; i < batches; i++) {
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    h_data[i].b[n][b] = 0;

            for (int blk = 0; blk < PARALLEL; blk++) {
                int global_idx = i * PARALLEL + blk;
                if (global_idx >= N) break;

                for (int n = 0; n < 16; n++) {
                    uint8_t val = global_idx % 16;  // same as CPU
                    for (int b = 0; b < 4; b++)
                        if (val & (1 << b))
                            h_data[i].b[n][b] |= (1u << blk);
                }
            }
        }

        // ---------------------------
        // Device allocation
        // ---------------------------
        BSState *d_data;
        cudaMalloc(&d_data, batches * sizeof(BSState));

        uint32_t (*d_round_keys)[16][4];
        cudaMalloc(&d_round_keys, sizeof(round_keys));
        cudaMemcpy(d_round_keys, round_keys, sizeof(round_keys), cudaMemcpyHostToDevice);

        // ---------------------------
        // Copy H->D (measure memory time)
        // ---------------------------
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem);
        cudaEventCreate(&stop_mem);

        cudaEventRecord(start_mem);
        cudaMemcpy(d_data, h_data.data(), batches * sizeof(BSState), cudaMemcpyHostToDevice);
        cudaEventRecord(stop_mem);
        cudaEventSynchronize(stop_mem);

        float mem_time = 0;
        cudaEventElapsedTime(&mem_time, start_mem, stop_mem);

        // ---------------------------
        // Kernel launch
        // ---------------------------
        int blockSize = 256;
        int gridSize = (batches + blockSize - 1) / blockSize;

        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        cudaEventRecord(start_kernel);
        twine_kernel<<<gridSize, blockSize>>>(d_data, batches, d_round_keys);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        // ---------------------------
        // Copy D->H only to read first block
        // ---------------------------
        cudaMemcpy(h_data.data(), d_data, batches * sizeof(BSState), cudaMemcpyDeviceToHost);

        // Print first block ciphertext
        std::cout << "Ciphertext (first block): ";
        for (int n = 0; n < 16; n++) {
            uint8_t val = 0;
            for (int b = 0; b < 4; b++)
                val |= ((h_data[0].b[n][b] & 1) << b);
            std::cout << std::hex << (int)val << " ";
        }
        std::cout << std::dec << "\n";

        double kernel_sec = kernel_time / 1000.0;
        double mem_sec = mem_time / 1000.0;
        double throughput = (N * 16.0) / (kernel_sec * 1e9);

        std::cout << "N = " << N
                  << " | Mem Time = " << mem_sec << " s"
                  << " | Kernel Time = " << kernel_sec << " s"
                  << " | Throughput = " << throughput << " GB/s\n";

        cudaFree(d_data);
        cudaFree(d_round_keys);
    }

    return 0;
}