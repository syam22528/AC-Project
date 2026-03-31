#include <iostream>
#include <stdint.h>
#include <cuda.h>

#define ROUNDS 44

// Z3 sequence (same as CPU)
const uint8_t Z3[62] = {
1,1,1,1,0,1,0,1,1,0,0,0,1,0,0,1,
0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,1,
1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,1,
0,0,1,1,0,0,0,1,1,1,1,0,1,0
};

// Put round keys in constant memory
__constant__ uint32_t d_round_keys[ROUNDS];

// Rotate left
__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// Rotate right (needed for key expansion)
inline uint32_t rotr(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
}

// SIMON function
__device__ __forceinline__ uint32_t f(uint32_t x) {
    return (rotl(x, 1) & rotl(x, 8)) ^ rotl(x, 2);
}

// ✅ SAME key expansion as CPU
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

// Kernel
__global__ void simon_kernel(uint32_t *left, uint32_t *right, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {

        uint32_t l = left[i];
        uint32_t r = right[i];

        #pragma unroll
        for (int j = 0; j < ROUNDS; j++) {
            uint32_t temp = l;
            l = r ^ f(l) ^ d_round_keys[j];
            r = temp;
        }

        left[i] = l;
        right[i] = r;
    }
}

int main() {

    int test_sizes[] = {
        1024, 16384, 65536, 262144, 1048576,
        4194304, 10485760, 52428800 , 104857600
    };

    // ✅ SAME KEY AS CPU
    uint32_t key[4] = {1, 2, 3, 4};
    uint32_t h_keys[ROUNDS];

    // ✅ Proper expansion
    key_expansion(key, h_keys);

    // Copy to constant memory
    cudaMemcpyToSymbol(d_round_keys, h_keys, sizeof(uint32_t) * ROUNDS);

    std::cout << "===== GPU PERFORMANCE =====\n";

    for (int N : test_sizes) {

        uint32_t *h_left = new uint32_t[N];
        uint32_t *h_right = new uint32_t[N];

        for (int i = 0; i < N; i++) {
            h_left[i] = i;
            h_right[i] = i ^ 0xdeadbeef;
        }

        uint32_t *d_left, *d_right;

        cudaMalloc(&d_left, N * sizeof(uint32_t));
        cudaMalloc(&d_right, N * sizeof(uint32_t));

        // ---- MEMORY TIME ----
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem);
        cudaEventCreate(&stop_mem);

        cudaEventRecord(start_mem);

        cudaMemcpy(d_left, h_left, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, h_right, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop_mem);
        cudaEventSynchronize(stop_mem);

        float mem_time = 0;
        cudaEventElapsedTime(&mem_time, start_mem, stop_mem);

        // ---- KERNEL TIME ----
        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        cudaEventRecord(start_kernel);

        simon_kernel<<<gridSize, blockSize>>>(d_left, d_right, N);

        cudaDeviceSynchronize();

        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        // ---- COPY BACK ----
        cudaMemcpy(h_left, d_left, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_right, d_right, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // ✅ SHOULD MATCH CPU NOW
        std::cout << std::hex;
        std::cout << "Ciphertext (first block): "
                  << h_left[0] << " " << h_right[0] << std::endl;
        std::cout << std::dec;

        double kernel_sec = kernel_time / 1000.0;
        double throughput = (N * 8.0) / (kernel_sec * 1e9);

        std::cout << "N = " << N
                  << " | Kernel Time = " << kernel_sec << " s"
                  << " | Mem Time = " << mem_time / 1000.0 << " s"
                  << " | Throughput = " << throughput << " GB/s\n";

        cudaFree(d_left);
        cudaFree(d_right);
        delete[] h_left;
        delete[] h_right;
    }

    return 0;
}