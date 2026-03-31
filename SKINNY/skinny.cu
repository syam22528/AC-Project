#include <iostream>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <chrono>

#define ROUNDS 32
#define PARALLEL 32   // 32 blocks per bitslice batch

// ---------------------------
// Bit-sliced state
// ---------------------------
struct BSState {
    uint32_t b[16][4]; // 16 nibbles × 4 bits (bit-sliced 32 blocks)
};

// ---------------------------
// Bit-sliced S-box (matches CPU)
// ---------------------------
__device__ inline void sbox_bitslice(uint32_t &b0, uint32_t &b1,
                                     uint32_t &b2, uint32_t &b3) {
    uint32_t t0 = b1 ^ b2;
    uint32_t t1 = b0 & b3;
    uint32_t t2 = b0 ^ b1;
    uint32_t t3 = b2 | t1;

    b3 ^= t0;
    b2 ^= t2;
    b1 ^= t3;
    b0 ^= b3;
}

__device__ inline void sub_cells(BSState &s) {
    #pragma unroll
    for (int i = 0; i < 16; i++)
        sbox_bitslice(s.b[i][0], s.b[i][1], s.b[i][2], s.b[i][3]);
}

// Permutation table
__device__ const int PERM[16] = {
    0, 5, 10, 15,
    4, 9, 14, 3,
    8, 13, 2, 7,
    12, 1, 6, 11
};

__device__ inline void permute(BSState &s) {
    BSState tmp;
    #pragma unroll
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            tmp.b[i][b] = s.b[PERM[i]][b];
    s = tmp;
}

__device__ inline void add_round_key(BSState &s, BSState &k) {
    for (int i = 0; i < 16; i++)
        for (int b = 0; b < 4; b++)
            s.b[i][b] ^= k.b[i][b];
}

// Minimal key schedule (matches CPU)
__device__ inline void key_schedule(BSState &k, int r) {
    for (int i = 0; i < 15; i++)
        k.b[i][0] = k.b[i + 1][0];
    k.b[15][0] ^= (r & 1);
}

// ---------------------------
// Encrypt one bitslice batch (32 blocks)
// ---------------------------
__device__ void skinny_bitslice_encrypt(BSState &state, BSState key) {
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(state);
        add_round_key(state, key);   // XOR all 4 bits per nibble
        permute(state);
        key_schedule(key, r);        // rotate only b[i][0]
    }
}

// ---------------------------
// GPU kernel
// ---------------------------
__global__ void skinny_kernel(BSState *d_data, int batches, BSState key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < batches; i += stride)
        skinny_bitslice_encrypt(d_data[i], key);
}

// ---------------------------
// Host
// ---------------------------
int main() {
    int test_sizes[] = {
        1024, 16384, 65536, 262144,
        1048576, 4194304, 10485760,
        52428800, 104857600
    };

    std::cout << "===== GPU BIT-SLICE SKINNY PERFORMANCE =====\n";

    for (int t = 0; t < sizeof(test_sizes)/sizeof(int); t++) {
        int N = test_sizes[t];
        int batches = (N + PARALLEL - 1) / PARALLEL;

        // Host bitslice data
        std::vector<BSState> h_data(batches);

        for (int i = 0; i < batches; i++)
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    h_data[i].b[n][b] = i + n + b; // same as CPU

        // Key
        BSState key = {};

        // Device allocation
        BSState *d_data;
        cudaMalloc(&d_data, batches * sizeof(BSState));

        BSState d_key = key;

        // ---- MEMORY TIME ----
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem);
        cudaEventCreate(&stop_mem);

        cudaEventRecord(start_mem);
        cudaMemcpy(d_data, h_data.data(), batches * sizeof(BSState), cudaMemcpyHostToDevice);
        cudaEventRecord(stop_mem);
        cudaEventSynchronize(stop_mem);

        float mem_time = 0;
        cudaEventElapsedTime(&mem_time, start_mem, stop_mem);

        // ---- KERNEL TIME ----
        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        int blockSize = 256;
        int gridSize = (batches + blockSize - 1) / blockSize;

        cudaEventRecord(start_kernel);
        skinny_kernel<<<gridSize, blockSize>>>(d_data, batches, d_key);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        // ---- COPY BACK ----
        cudaMemcpy(h_data.data(), d_data, batches * sizeof(BSState), cudaMemcpyDeviceToHost);

        // Print first block ciphertext (matches CPU)
        std::cout << "Ciphertext (first block approx): ";
        for (int n = 0; n < 16; n++) {
            uint8_t val = 0;
            for (int b = 0; b < 4; b++)
                val |= ((h_data[0].b[n][b] & 1) << b);
            std::cout << std::hex << (int)val << " ";
        }
        std::cout << std::dec << "\n";

        double kernel_sec = kernel_time / 1000.0;
        double throughput = (N * 16.0) / (kernel_sec * 1e9);
        std::cout << "N = " << N
                  << " | Kernel Time = " << kernel_sec << " s"
                  << " | Mem Time = " << mem_time / 1000.0 << " s"
                  << " | Throughput = " << throughput << " GB/s\n";

        cudaFree(d_data);
    }

    return 0;
}