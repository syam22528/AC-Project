#include <iostream>
#include <cstdint>
#include <chrono>
using namespace std;

#define ROUNDS 36
#define N (1<<20)

static const uint8_t SBOX[16] = {
    0xC,0x0,0xF,0xA,0x2,0xB,0x9,0x5,
    0x8,0x3,0xD,0x7,0x1,0xE,0x6,0x4
};

static const uint8_t P[16] = {
    0,9,2,13,4,11,6,15,8,1,10,3,12,5,14,7
};

void encrypt(uint8_t *state) {
    uint8_t tmp[16];

    for (int r = 0; r < ROUNDS; r++) {
        // Substitute
        for (int i = 0; i < 16; i++)
            state[i] = SBOX[state[i]];

        // Permute
        for (int i = 0; i < 16; i++)
            tmp[i] = state[P[i]];

        for (int i = 0; i < 16; i++)
            state[i] = tmp[i];
    }
}

int main() {
    uint8_t *data = new uint8_t[N * 16];

    for (int i = 0; i < N * 16; i++)
        data[i] = i % 16;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++)
        encrypt(&data[i * 16]);

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();
    double throughput = (double)(N * 16) / time / 1e9; // GB/s

    cout << "Time: " << time << " s\n";
    cout << "Throughput: " << throughput << " GB/s\n";

    delete[] data;
}