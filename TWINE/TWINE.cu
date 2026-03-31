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
// Bit-sliced TWINE S-box
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
// Permutation table
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
// Encrypt one bitslice batch
// ---------------------------
__device__ void twine_bitslice_encrypt(BSState &s) {
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(s);
        permute(s);
    }
}

// ---------------------------
// GPU kernel
// ---------------------------
__global__ void twine_kernel(BSState *d_data, int batches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < batches; i += stride)
        twine_bitslice_encrypt(d_data[i]);
}

// ---------------------------
// Host
// ---------------------------
int main() {
    int test_sizes[] = {1024, 16384, 65536, 262144,
                        1048576, 4194304, 10485760,
                        52428800, 104857600};

    std::cout << "===== GPU BIT-SLICE TWINE PERFORMANCE =====\n";

    for (int t = 0; t < sizeof(test_sizes)/sizeof(int); t++) {
        int N = test_sizes[t];
        int batches = (N + PARALLEL - 1) / PARALLEL;

        // Host bitslice data
        std::vector<BSState> h_data(batches);

        for (int i = 0; i < batches; i++)
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    h_data[i].b[n][b] = i + n + b;

        // Device allocation
        BSState *d_data;
        cudaMalloc(&d_data, batches * sizeof(BSState));

        // ---- MEMORY H->D TIME ----
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
        int blockSize = 256;
        int gridSize = (batches + blockSize - 1) / blockSize;

        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        cudaEventRecord(start_kernel);
        twine_kernel<<<gridSize, blockSize>>>(d_data, batches);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        // Copy back only to read first block ciphertext
        cudaMemcpy(h_data.data(), d_data, batches * sizeof(BSState), cudaMemcpyDeviceToHost);

        // Print first block ciphertext (approx)
        std::cout << "Ciphertext (first block approx): ";
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
    }

    return 0;
}