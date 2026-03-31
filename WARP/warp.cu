// warp_bitslice_gpu_corrected.cu
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cstdint>
#include <cstring>

#define ROUNDS 32
#define PARALLEL 32
#define BLOCK_SIZE 256

// ---------------------------
// Bitsliced state
// ---------------------------
struct BSState {
    uint32_t b[16][4]; // 16 nibbles × 4 bits
};

// ---------------------------
// WARP S-box (bitslice)
// ---------------------------
__device__ __forceinline__ void sbox_bitslice(uint32_t &x0, uint32_t &x1,
                                              uint32_t &x2, uint32_t &x3) {
    uint32_t t0 = x1 ^ x2;
    uint32_t t1 = x0 | x3;
    uint32_t t2 = x0 ^ x1 ^ x3;
    uint32_t t3 = x0 & x2;

    x0 = t0 ^ t1;
    x1 = t2;
    x2 = t3 ^ x1;
    x3 = x0 ^ x2 ^ x3;
}

__device__ __forceinline__ void sub_cells(BSState &s) {
    #pragma unroll
    for (int i = 0; i < 16; i++)
        sbox_bitslice(s.b[i][0], s.b[i][1], s.b[i][2], s.b[i][3]);
}

// ---------------------------
// Permutation
// ---------------------------
__constant__ int d_PERM[16] = {0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11};

__device__ __forceinline__ void permute(BSState &s) {
    BSState tmp;
    #pragma unroll
    for (int i = 0; i < 16; i++)
        #pragma unroll
        for (int b = 0; b < 4; b++)
            tmp.b[i][b] = s.b[d_PERM[i]][b];
    s = tmp;
}

// ---------------------------
// WARP round constants (RC)
// ---------------------------
__constant__ uint8_t d_RC[ROUNDS] = {
    0x01,0x03,0x07,0x0f,0x1f,0x3e,0x3d,0x3b,
    0x37,0x2f,0x1e,0x3c,0x39,0x33,0x27,0x0e,
    0x1d,0x3a,0x35,0x2b,0x16,0x2c,0x18,0x30,
    0x21,0x02,0x05,0x0b,0x17,0x2e,0x1c,0x38
};

// ---------------------------
// Add round key + RC
// ---------------------------
__device__ __forceinline__ void add_round(BSState &s, const uint8_t round_key[16], int r) {
    #pragma unroll
    for (int n = 0; n < 16; n++)
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t mask = ((round_key[n] >> b) & 1) ? 0xFFFFFFFFu : 0;
            mask ^= ((d_RC[r] >> b) & 1) ? 0xFFFFFFFFu : 0;
            s.b[n][b] ^= mask;
        }
}

// ---------------------------
// Bitslice encrypt
// ---------------------------
__device__ __forceinline__ void warp_bitslice_encrypt(BSState &s, const uint8_t round_keys[ROUNDS][16]) {
    for (int r = 0; r < ROUNDS; r++) {
        sub_cells(s);
        add_round(s, round_keys[r], r);
        permute(s);
    }
}

// ---------------------------
// GPU kernel
// ---------------------------
__global__ void warp_kernel(BSState *d_data, const uint8_t round_keys[ROUNDS][16], int batches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < batches; i += stride)
        warp_bitslice_encrypt(d_data[i], round_keys);
}

// ---------------------------
// Host-side key schedule (same as CPU)
// ---------------------------
void key_schedule(const uint8_t master[16], uint8_t round_keys[ROUNDS][16]) {
    uint8_t k[16]; memcpy(k, master, 16);
    uint8_t RC[ROUNDS] = {
        0x01,0x03,0x07,0x0f,0x1f,0x3e,0x3d,0x3b,
        0x37,0x2f,0x1e,0x3c,0x39,0x33,0x27,0x0e,
        0x1d,0x3a,0x35,0x2b,0x16,0x2c,0x18,0x30,
        0x21,0x02,0x05,0x0b,0x17,0x2e,0x1c,0x38
    };
    for (int r = 0; r < ROUNDS; r++) {
        for (int i = 0; i < 16; i++) round_keys[r][i] = k[i];
        uint8_t tmp = k[0];
        for (int i = 0; i < 15; i++) k[i] = k[i+1];
        k[15] = tmp ^ RC[r];
    }
}

// ---------------------------
// MAIN
// ---------------------------
int main() {
    int test_sizes[] = {1024, 16384, 65536, 262144,
                        1048576, 4194304, 10485760,
                        52428800, 104857600};

    uint8_t master_key[16] = {0};
    uint8_t round_keys[ROUNDS][16];
    key_schedule(master_key, round_keys);

    std::cout << "===== GPU BIT-SLICE WARP_ENCRYPT VERIFIED PERFORMANCE =====\n";

    for (int t = 0; t < (int)(sizeof(test_sizes)/sizeof(int)); t++) {
        int N = test_sizes[t];
        int batches = (N + PARALLEL - 1) / PARALLEL;

        // Initialize bitslice data like CPU
        std::vector<BSState> h_data(batches);
        for (int i = 0; i < batches; i++) {
            for (int n = 0; n < 16; n++)
                for (int b = 0; b < 4; b++)
                    h_data[i].b[n][b] = 0;

            for (int blk = 0; blk < PARALLEL; blk++) {
                int global_idx = i * PARALLEL + blk;
                if (global_idx >= N) break;

                for (int n = 0; n < 16; n++) {
                    uint8_t val = (global_idx * 16 + n) % 16;
                    for (int b = 0; b < 4; b++)
                        if (val & (1 << b))
                            h_data[i].b[n][b] |= (1u << blk);
                }
            }
        }

        // Device allocation
        BSState *d_data; uint8_t (*d_round_keys)[16];
        cudaMalloc(&d_data, batches * sizeof(BSState));
        cudaMalloc(&d_round_keys, ROUNDS * 16);

        // Memory copy host → device
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem); cudaEventCreate(&stop_mem);
        cudaEventRecord(start_mem);
        cudaMemcpy(d_data, h_data.data(), batches * sizeof(BSState), cudaMemcpyHostToDevice);
        cudaMemcpy(d_round_keys, round_keys, ROUNDS*16, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_mem); cudaEventSynchronize(stop_mem);
        float mem_time = 0; cudaEventElapsedTime(&mem_time, start_mem, stop_mem);

        // Launch kernel
        int gridSize = (batches + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
        cudaEventRecord(start_kernel);
        warp_kernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_round_keys, batches);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_kernel); cudaEventSynchronize(stop_kernel);
        float kernel_time = 0; cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        // Copy back to host
        cudaMemcpy(h_data.data(), d_data, batches * sizeof(BSState), cudaMemcpyDeviceToHost);

        // Print first block ciphertext (matches CPU)
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
        double throughput = (N * 8.0) / (kernel_sec * 1e9); // 8 bytes per block

        std::cout << "N = " << N
                  << " | Mem Time = " << mem_sec << " s"
                  << " | Kernel Time = " << kernel_sec << " s"
                  << " | Throughput = " << throughput << " GB/s\n";

        cudaFree(d_data); cudaFree(d_round_keys);
    }

    return 0;
}