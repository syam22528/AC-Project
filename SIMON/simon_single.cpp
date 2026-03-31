#include <iostream>
#include <cstdint>
#include <chrono>

using namespace std;

#define ROUNDS 44

inline uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t f(uint32_t x) {
    return (rotl(x, 1) & rotl(x, 8)) ^ rotl(x, 2);
}

void simon_encrypt(uint32_t &l, uint32_t &r, uint32_t round_keys[ROUNDS]) {
    for (int i = 0; i < ROUNDS; i++) {
        uint32_t temp = l;
        l = r ^ f(l) ^ round_keys[i];
        r = temp;
    }
}

int main() {

    // 🔹 Test input
    uint32_t left = 0x656b696c;
    uint32_t right = 0x20646e75;

    uint32_t round_keys[ROUNDS];

    // simple test keys
    for (int i = 0; i < ROUNDS; i++)
        round_keys[i] = i;

    // 🔹 Number of repetitions (increase for stable timing)
    const int N = 4194304;  // 10 million encryptions

    uint32_t l, r;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        l = left;
        r = right;
        simon_encrypt(l, r, round_keys);
    }

    auto end = chrono::high_resolution_clock::now();

    double time = chrono::duration<double>(end - start).count();

    // 64-bit block = 8 bytes
    double throughput = (N * 8.0) / (time * 1e9);

    // 🔹 Print final result (correctness)
    cout << "Output: " << hex << l << " " << r << endl;

    // 🔹 Timing
    cout << dec;
    cout << "Time: " << time << " s\n";
    cout << "Throughput: " << throughput << " GB/s\n";

    return 0;
}