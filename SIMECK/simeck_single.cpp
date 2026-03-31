#include <iostream>
#include <cstdint>
#include <chrono>
using namespace std;

#define ROUNDS 44
#define N (1<<20)

// Force inline for performance
inline __attribute__((always_inline)) uint32_t ROL(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline __attribute__((always_inline)) uint32_t f(uint32_t x) {
    return (ROL(x, 5) & x) ^ ROL(x, 1);
}

// Encrypt one block
inline __attribute__((always_inline)) void encrypt(uint32_t &x, uint32_t &y, const uint32_t *rk) {
    #pragma unroll
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t tmp = x;
        x = y ^ f(x) ^ rk[i];
        y = tmp;
    }
}

int main() {

    uint32_t *x = new uint32_t[N];
    uint32_t *y = new uint32_t[N];
    uint32_t rk[ROUNDS];

    // Init (same pattern as GPU-style)
    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = i + 1;
    }

    // Simple round keys
    for (int i = 0; i < ROUNDS; i++)
        rk[i] = i;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        encrypt(x[i], y[i], rk);
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double>(end - start).count();
    double throughput = (double)(N * 8) / time / 1e9;

    // ✅ PRINT CIPHERTEXT (for correctness)
    cout << hex;
    cout << "Ciphertext (first block): "
         << x[0] << " " << y[0] << endl;
    cout << dec;

    cout << "Time: " << time << " s\n";
    cout << "Throughput: " << throughput << " GB/s\n";

    delete[] x;
    delete[] y;

    return 0;
}