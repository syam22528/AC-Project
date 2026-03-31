#include <iostream>
#include <stdint.h>
#include <cuda.h>

#define ROUNDS 44

// Z3 sequence (LSB first)
const uint8_t Z3[62] = {
1,1,1,1,0,1,0,1,1,0,0,0,1,0,0,1,
0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,1,
1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,1,
0,0,1,1,0,0,0,1,1,1,1,0,1,0
};

// Round keys in constant memory
__constant__ uint32_t d_round_keys[ROUNDS];

// Rotate left
__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// Rotate right (host only, for key expansion)
inline uint32_t rotr(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
}

// SIMON round function
__device__ __forceinline__ uint32_t f(uint32_t x) {
    return (rotl(x,1) & rotl(x,8)) ^ rotl(x,2);
}

// ✅ Correct CPU key expansion for SIMON 64/128
void key_expansion(const uint32_t key[4], uint32_t round_keys[ROUNDS]) {
    const uint32_t c = 0xfffffffc;

    // Copy master key
    for(int i=0;i<4;i++)
        round_keys[i] = key[i];

    for(int i=4;i<ROUNDS;i++) {
        uint32_t tmp = rotr(round_keys[i-1],3) ^ round_keys[i-3];
        tmp ^= rotr(tmp,1);
        uint32_t z_bit = Z3[(i-4)%62];
        round_keys[i] = c ^ z_bit ^ tmp ^ round_keys[i-4]; // ✅ NO ~
    }
}

// Kernel: encrypt many blocks in parallel
__global__ void simon_kernel(uint32_t *left, uint32_t *right, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=idx;i<N;i+=stride) {
        uint32_t l = left[i];
        uint32_t r = right[i];

        #pragma unroll
        for(int j=0;j<ROUNDS;j++) {
            uint32_t tmp = l;
            l = r ^ f(l) ^ d_round_keys[j];
            r = tmp;
        }

        left[i] = l;
        right[i] = r;
    }
}

int main() {
    uint32_t key[4] = {0x19181110,0x11100908,0x09050302,0x01000000};
    uint32_t round_keys[ROUNDS];

    key_expansion(key, round_keys);
    cudaMemcpyToSymbol(d_round_keys, round_keys, sizeof(uint32_t)*ROUNDS);

    int test_sizes[] = {1024, 16384, 65536, 262144, 1048576, 4194304, 10485760};

    std::cout << "===== GPU PERFORMANCE =====\n";

    for(int N : test_sizes) {
        uint32_t *h_left = new uint32_t[N];
        uint32_t *h_right = new uint32_t[N];

        for(int i=0;i<N;i++){
            h_left[i] = i;
            h_right[i] = i ^ 0xabcdabcd;
        }

        uint32_t *d_left, *d_right;
        cudaMalloc(&d_left, N*sizeof(uint32_t));
        cudaMalloc(&d_right, N*sizeof(uint32_t));

        // --- MEMORY TIMING START ---
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem);
        cudaEventCreate(&stop_mem);
        cudaEventRecord(start_mem);

        cudaMemcpy(d_left, h_left, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, h_right, N*sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop_mem);
        cudaEventSynchronize(stop_mem);
        float mem_ms = 0;
        cudaEventElapsedTime(&mem_ms, start_mem, stop_mem);
        // --- MEMORY TIMING END ---

        // --- KERNEL TIMING START ---
        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);
        cudaEventRecord(start_kernel);

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        simon_kernel<<<gridSize, blockSize>>>(d_left, d_right, N);

        cudaDeviceSynchronize();
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
        // --- KERNEL TIMING END ---

        // Copy back to host
        cudaMemcpy(h_left, d_left, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_right, d_right, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // ✅ Verification: first block
        std::cout << std::hex;
        std::cout << "Ciphertext (first block): " << h_left[0] << " " << h_right[0] << std::endl;
        std::cout << std::dec;

        double kernel_sec = kernel_ms / 1000.0;
        double mem_sec = mem_ms / 1000.0;
        double throughput = (N * 8.0) / (kernel_sec * 1e9); // GB/s, kernel only

        std::cout << "N = " << N
                  << " | Kernel Time = " << kernel_sec << " s"
                  << " | Mem Time = " << mem_sec << " s"
                  << " | Throughput = " << throughput << " GB/s\n";

        cudaFree(d_left);
        cudaFree(d_right);
        delete[] h_left;
        delete[] h_right;
    }

    return 0;
}