#include <iostream>
#include <cstdint>
#include <chrono>
using namespace std;

#define ROUNDS 32
#define N (1<<20)

// SBOX
static const uint8_t SBOX[16] = {
    0xC,0x6,0x9,0x0,0x1,0xA,0x2,0xB,
    0x3,0x8,0x5,0xD,0x4,0xE,0x7,0xF
};

// PERM
static const uint8_t PERM[16] = {
    0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11
};

inline void sub_cells(uint8_t *s){
    for(int i=0;i<16;i++) s[i]=SBOX[s[i]&0xF];
}

inline void permute(uint8_t *s){
    uint8_t t[16];
    for(int i=0;i<16;i++) t[i]=s[PERM[i]];
    for(int i=0;i<16;i++) s[i]=t[i];
}

inline void add_round_key(uint8_t *s,uint8_t *k){
    for(int i=0;i<16;i++) s[i]^=k[i];
}

inline void key_schedule(uint8_t *k,int r){
    uint8_t tmp=k[0];
    for(int i=0;i<15;i++) k[i]=k[i+1];
    k[15]=tmp^(r&0xF);
}

inline void encrypt(uint8_t *s,const uint8_t *master){
    uint8_t k[16];
    for(int i=0;i<16;i++) k[i]=master[i];

    for(int r=0;r<ROUNDS;r++){
        sub_cells(s);
        add_round_key(s,k);
        permute(s);
        key_schedule(k,r);
    }
}

int main(){
    uint8_t *data=new uint8_t[N*16];
    uint8_t key[16]={0};

    for(int i=0;i<N;i++)
        for(int j=0;j<16;j++)
            data[i*16+j]=(i+j)&0xFF;

    auto t1=chrono::high_resolution_clock::now();

    for(int i=0;i<N;i++)
        encrypt(&data[i*16],key);

    auto t2=chrono::high_resolution_clock::now();

    double time=chrono::duration<double>(t2-t1).count();
    double th=(N*16.0)/(time*1e9);

    cout<<"Ciphertext: ";
    for(int i=0;i<16;i++) cout<<hex<<(int)data[i]<<" ";
    cout<<dec<<"\nTime: "<<time<<" s\nThroughput: "<<th<<" GB/s\n";
}