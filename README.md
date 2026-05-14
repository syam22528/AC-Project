# ⚡ GPU-Accelerated Block Ciphers: Performance Analysis of ARX, Feistel, and SPN Architectures

> **Systematic implementation, optimization, and benchmarking of 12 block ciphers across three cryptographic families on CPU and GPU (CUDA).**

**Authors:** Armaan Singh, Syam Sai Santosh, Bhargav Raman

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Cipher Families](#cipher-families)
  - [ARX-Based Ciphers](#1-arx-based-ciphers)
  - [Feistel-Based Ciphers](#2-feistel-based-ciphers-lightweight)
  - [SPN-Based Ciphers](#3-spn-based-ciphers-standard--advanced)
- [Project Structure](#project-structure)
- [Hardware & Software Setup](#hardware--software-setup)
- [CPU Implementation](#cpu-implementation)
- [GPU Implementation (CUDA)](#gpu-implementation-cuda)
- [GPU Optimization Techniques](#gpu-optimization-techniques)
- [Benchmark Results Summary](#benchmark-results-summary)
- [Correctness Verification](#correctness-verification)
- [Reproducing Results](#reproducing-results)
- [Reports & Documentation](#reports--documentation)

---

## Abstract

Modern cryptographic systems must handle large volumes of data efficiently. Block ciphers, which operate on fixed-size data blocks, are naturally parallelizable since independent blocks can be encrypted simultaneously. This makes them well-suited for acceleration using Graphics Processing Units (GPUs).

This project implements, optimizes, and benchmarks **twelve symmetric ciphers** across ARX, Feistel, and Substitution-Permutation Network (SPN) architectures. We evaluate each cipher across multiple implementation tiers — ranging from single-threaded baselines to heavily optimized CUDA GPU kernels — to analyze performance bottlenecks, memory transfer overheads, and the architectural factors that influence GPU speedup.

---

## Cipher Families

### 1. ARX-Based Ciphers

Located in [`ARX_based/`](ARX_based/)

| Cipher | Block Size | Key Size | Rounds | Structure |
|--------|-----------|----------|--------|-----------|
| **SPECK 128/128** | 128-bit | 128-bit | 32 | ARX (Feistel-like) |
| **LEA-128** | 128-bit | 128-bit | 24 | ARX (Generalized Feistel Network) |
| **CHAM-128/128** | 128-bit | 128-bit | 80 | ARX (compact key schedule) |
| **Threefish-256** | 256-bit | 256-bit | 72 | Pure ARX (no S-boxes, used in Skein hash) |

**Key highlights:**
- All four implementations share a consistent Python + Numba CUDA architecture
- GPU kernels use **shared memory** for round keys and **grid-stride loops** for arbitrary input sizes
- Peak GPU throughput converges to **~550 MB/s** across all ciphers, revealing PCIe transfer as the bottleneck
- Best speedup: **119× (SPECK at 10 MB)**, **93.6× sustained (LEA at 100 MB)**

---

### 2. Feistel-Based Ciphers (Lightweight)

Located in [`Fiestel_based/`](Fiestel_based/)

| Cipher | Block Size | Key Size | Rounds | Structure |
|--------|-----------|----------|--------|-----------|
| **SIMECK 64/128** | 64-bit | 128-bit | 44 | ARX-based Feistel (rotation, AND, XOR) |
| **SIMON 64/128** | 64-bit | 128-bit | 44 | Feistel (rotation-AND-XOR) |
| **TWINE** | 64-bit | 80-bit | 36 | Generalized Feistel (4-bit nibble S-box) |
| **WARP** | 128-bit | 128-bit | 41 | Substitution-Permutation with linear diffusion |

**Key highlights:**
- Implemented in **C/C++ (OpenMP)** for CPU and **CUDA C** for GPU
- **Bit-slicing optimization** used for TWINE and WARP (32 blocks processed per thread)
- **Highest speedups of all 12 ciphers**: SIMECK achieves **~800×** and SIMON achieves **~550×** — their simple bitwise round functions map directly to fast GPU instructions with minimal register pressure
- TWINE bit-sliced reaches **~60×** speedup; WARP achieves **~7×**

---

### 3. SPN-Based Ciphers (Standard & Advanced)

Located in [`SPN_based/`](SPN_based/)

| Cipher | Block Size | Key Size | Rounds | State Representation |
|--------|-----------|----------|--------|---------------------|
| **AES-128** | 128-bit | 128-bit | 10 | Column-major byte array `uint8[16]` |
| **GIFT-64-128** | 64-bit | 128-bit | 28 | Packed `uint64` |
| **SKINNY-64-128** | 64-bit | 128-bit | 36 | Packed `uint64` (4×16-bit rows) |
| **PRESENT-128** | 64-bit | 128-bit | 31 | Packed `uint64` |

**Key highlights:**
- Each cipher offers **multiple GPU kernel variants** (table-based vs. bitsliced)
- Advanced optimizations: thread coarsening (2–8 blocks/thread), fused SP scatter tables, delta-swap permutations, pinned host memory
- Both **ECB** and **CTR** modes benchmarked
- **Peak GPU throughput: 1,346 MB/s (AES)**; **Peak CTR speedup: 14.21× (PRESENT)**
- Bitsliced variants drastically outperform table lookups — SKINNY bitsliced peaks at **972 MB/s**, nearly matching AES

---

## Project Structure

```
AC-Project/
│
├── README.md                          ← This file
├── GPU_Accelerated_Block_Ciphers_Report.pdf  ← Full project report
│
├── ARX_based/                         ← ARX cipher implementations (Python + Numba CUDA)
│   ├── SPECK/
│   │   ├── speck_naive.py             # Pure Python reference
│   │   ├── speck_optimized.py         # Batch-processed Python
│   │   ├── speck_numba.py             # Numba JIT (CPU)
│   │   ├── speck_gpu.py              # CUDA kernel
│   │   ├── benchmark_speck.py         # Benchmark harness
│   │   └── test_speck.py              # Correctness tests
│   ├── LEA/                           # Same structure as SPECK
│   ├── CHAM/                          # Same structure as SPECK
│   ├── THREEFISH/                     # Same structure as SPECK
│   ├── Plots/                         # Generated benchmark plots (PNG)
│   ├── main.py                        # Entry point: benchmarks + plot generation
│   ├── output.txt                     # Sample benchmark output
│   ├── requirements.txt               # Python dependencies
│   └── report.md                      # Detailed ARX analysis report
│
├── Fiestel_based/                     ← Feistel cipher implementations (C/C++ / CUDA C)
│   ├── SIMON/
│   │   ├── simon.cpp                  # CPU (OpenMP multi-threaded)
│   │   ├── simon_single.cpp           # CPU (single-threaded baseline)
│   │   ├── simon.cu                   # CUDA GPU kernel
│   │   └── simon.py                   # Plotting / visualization script
│   ├── SIMECK/                        # Same structure as SIMON
│   ├── TWINE/                         # Same structure (+ bit-slicing)
│   ├── WARP/                          # Same structure (+ bit-slicing)
│   └── report.tex                     # LaTeX report for Feistel ciphers
│
├── SPN_based/                         ← SPN cipher implementations (Python + Numba CUDA)
│   ├── aes/
│   │   ├── cpu.py                     # Numba JIT AES-128 (ECB + CTR)
│   │   ├── gpu.py                     # GPU: shared S-box, global/constkeys modes
│   │   ├── benchmark.py               # Benchmark driver
│   │   └── verify.py                  # NIST test vector verification
│   ├── gift/                          # Table + bitsliced GPU variants
│   ├── present/                       # Bitsliced + table variants, native CTR kernel
│   ├── skinny/                        # Dual-nibble SBOX8 + bitsliced variants
│   ├── results/                       # Benchmark CSVs + generated plots
│   ├── ctr_utils.py                   # Shared CTR-mode utilities
│   ├── plot_mbps_graphs.py            # Plot generation script
│   └── SPN_CIPHER_OPTIMIZATIONS.md    # Detailed SPN optimization report
```

---

## Hardware & Software Setup

Benchmarks were conducted across three separate systems due to the varied nature of the cipher stacks:

| System | Used For | CPU | GPU | VRAM | RAM | CUDA |
|--------|----------|-----|-----|------|-----|------|
| **A** | ARX ciphers | Intel Ultra 7 155H | NVIDIA RTX 4060 Laptop (65W) | 6 GB | 16 GB | 11.8 |
| **B** | Feistel ciphers | AMD Ryzen 7 5800H | NVIDIA RTX 3050 Ti (95W) | — | 16 GB | 12.3 |
| **C** | SPN ciphers | Intel i7 11370H | NVIDIA RTX 3050 Ti (50W) | 4 GB | 16 GB | 12.8 |

**Additional software:**
- **ARX & SPN:** Python 3.10+, Numba CUDA JIT, NumPy, Matplotlib
- **Feistel:** Visual Studio 2022 (MSVC v143), OpenMP

Input sizes swept from N = 2¹⁰ (1 KB) up to 10⁸ blocks (~1.6 GB).

---

## CPU Implementation

### Single-Threaded & Python Reference Baselines

- **ARX ciphers**: Pure Python implementations processed one block at a time, then batched to minimize parsing overhead.
- **Feistel ciphers**: Correct reference implementations developed in C/C++ based on standard specifications.
- Both serve as **correctness references** and **absolute baselines**.

### Multi-Threaded & Compiled CPU Execution

| Strategy | Ciphers | Description |
|----------|---------|-------------|
| **Numba JIT (Python)** | SPECK, LEA, CHAM, Threefish | NumPy arrays with `@njit` compiled loops. |
| **Numba JIT (Python)** | AES, GIFT, SKINNY, PRESENT | `@njit(cache=True)` with OpenMP `prange` for parallel block processing across 8 CPU threads. |
| **OpenMP (C/C++)** | SIMECK, SIMON, TWINE, WARP | `#pragma omp parallel for` across plaintext blocks. |

### Bit-Slicing Optimization (CPU)

For **TWINE, WARP, SKINNY, and PRESENT**, bit-sliced implementations store bits across 32-bit or 64-bit words, enabling SIMD-style parallelism on the CPU. This processes 32 blocks simultaneously and heavily reduces memory overhead.

---

## GPU Implementation (CUDA)

### Parallelization Strategy & Thread Mapping

All GPU implementations use **grid-stride loops** for clean handling of arbitrarily large inputs:

```c
for (int i = idx; i < N; i += stride)
    encrypt(...)
```

Thread mapping varies based on arithmetic intensity:

| Strategy | Ciphers | Blocks/Thread |
|----------|---------|---------------|
| Standard (1 block/thread) | SIMECK, SIMON, SPECK, LEA, CHAM | 1 |
| Bit-sliced batch | TWINE, WARP | 32 |
| Thread coarsening | AES (2×), GIFT/SKINNY (4×), PRESENT (8×) | 2–8 |

All use **256 threads per block** aligned with warp scheduling for high occupancy.

### Memory Management

| Memory Tier | Usage | Ciphers |
|-------------|-------|---------|
| **Global** | Plaintext/ciphertext buffers (≥1 GB) | All |
| **Shared** | AES S-box (256 B), round key staging | AES, GIFT, SKINNY, PRESENT |
| **Constant** | Round keys, S-boxes (16 B), permutation tables | SIMECK, SIMON, GIFT, SKINNY, PRESENT |
| **Registers** | Temporary variables during rounds | All |

### Mode of Operation: ECB vs. CTR

- **ECB**: Independent parallel encryption of each block — best GPU throughput.
- **CTR**: Introduces counter construction and XOR overhead. For **PRESENT**, a dedicated **native CTR kernel** fuses counter generation, encryption, and XOR entirely on-device, bypassing host-side bottlenecks.

---

## GPU Optimization Techniques

### S-Box & Linear Layer Optimizations

| Technique | Cipher | Impact |
|-----------|--------|--------|
| **Branchless `xtime` MixColumns** | AES | Avoids divergent warps via compound-XOR tricks. |
| **Delta-swap pLayer** | PRESENT | 4 operations replace a 63-iteration bit loop (20 vs 189 instructions). |
| **Bitsliced Boolean S-boxes** | SKINNY, PRESENT | Zero-memory-latency logic gate implementations; 4–12× faster kernel than table. |
| **Fused SP scatter table** | GIFT | 256-entry constant-memory table replaces 80 loop iterations with 16 lookups. |
| **Dual-nibble SBOX8** | SKINNY | Halves S-box memory operations (8 lookups vs 16). |

### Kernel-Level Optimizations

| Technique | Cipher | Impact |
|-----------|--------|--------|
| Thread coarsening (2–8 blocks/thread) | AES, GIFT, SKINNY, PRESENT | Amortizes shared-memory setup overhead. |
| Shared-memory S-box (256 B) | AES | Eliminates 160 global-memory lookups per block. |
| Shared-memory round-key staging | GIFT, SKINNY, PRESENT | Single load per block, reused across all rounds. |
| Constant-memory round keys (constkeys) | AES | Key-specialized kernel; 1,265 MB/s vs 1,015 MB/s at peak. |
| Pinned host memory | PRESENT | Avoids repeated page-locking overhead. |
| Occupancy-aware block sizing | PRESENT | Maximizes SM utilization via CUDA occupancy API. |
| Native CTR kernel | PRESENT | Fuses counter + encrypt + XOR; achieves **14.21× CTR speedup**. |

---

## Benchmark Results Summary

### ARX Ciphers — GPU Throughput @ 100 MB

| Cipher | GPU Throughput | Numba Throughput | Speedup |
|--------|---------------|-----------------|---------|
| **SPECK** | 571.67 MB/s | 15.49 MB/s | 36.91× |
| **LEA** | 581.65 MB/s | 6.21 MB/s | 93.61× |
| **CHAM** | 563.10 MB/s | 196.16 MB/s | 2.87× |
| **Threefish** | 530.83 MB/s | 178.73 MB/s | 2.97× |

> All ARX ciphers converge to ~530–580 MB/s GPU throughput. Memory transfer accounts for **78–87%** of total GPU time at 100 MB.

### Feistel Ciphers — Peak GPU Performance

| Cipher | Peak GPU Throughput | Peak Speedup | Notes |
|--------|-------------------|-------------|-------|
| **SIMECK** | ~70 GB/s | **~800×** | Simple rotation-AND-XOR maps perfectly to GPU |
| **SIMON** | ~40 GB/s | **~550×** | Similar structure to SIMECK |
| **TWINE** (bit-sliced) | ~27 GB/s | **~60×** | 32 blocks/thread via bit-slicing |
| **WARP** (bit-sliced) | ~0.38 GB/s | **~7×** | Complex linear diffusion limits speedup |

> SIMECK and SIMON achieve the **highest speedups among all 12 ciphers** — their extremely simple round functions map directly to fast GPU bitwise instructions with minimal register pressure.

### SPN Ciphers — Peak GPU Performance

| Cipher | Best Variant | Peak Throughput | Peak ECB Speedup | Peak CTR Speedup |
|--------|-------------|----------------|-----------------|-----------------|
| **AES-128** | constkeys | **1,346 MB/s** | 9.90× | 3.30× |
| **GIFT-64-128** | table | 328 MB/s | 6.12× | 5.13× |
| **SKINNY-64-128** | bitsliced | 972 MB/s | 11.22× | 6.44× |
| **PRESENT-128** | bitsliced | 795 MB/s | 11.29× | **14.21×** |

> AES dominates absolute throughput (128-bit blocks move 2× more data), but PRESENT achieves the highest CTR speedup via its native fused kernel.

### Cross-Family Peak Performance

| Metric | Winner | Value |
|--------|--------|-------|
| Highest GPU speedup overall | SIMECK (Feistel) | **~800×** |
| Highest absolute throughput | AES (SPN, constkeys, ECB) | **1,346 MB/s** |
| Highest ECB speedup (SPN) | PRESENT (bitsliced) | **11.29×** |
| Highest CTR speedup | PRESENT (native CTR) | **14.21×** |
| Highest kernel-only speedup | AES (constkeys) | **170.9×** |
| Highest sustained ARX speedup | LEA (100 MB) | **93.6×** |
| Peak ARX speedup | SPECK (10 MB) | **119×** |
| Fastest lightweight SPN cipher | SKINNY (bitsliced) | **972 MB/s** |
| Most transfer-dominated | AES | **92% transfer** |
| Most compute-dominated | GIFT (table) | **72% compute** |

### Key Cross-Family Observations

- **PCIe transfer is the universal bottleneck**: ARX ciphers at 78–87% transfer at 100 MB, AES at 90% at 4M blocks.
- **Simple round functions yield highest speedups**: SIMECK/SIMON (550–800×) >> SPN ciphers (~11×) >> complex ARX ciphers (2–3×).
- **Bit-slicing is highly effective on GPU**: 3–12× better kernel performance than table-based approaches across TWINE, WARP, SKINNY, and PRESENT.
- **Speedup depends on CPU baseline quality**: Ciphers with slow CPU implementations (LEA, PRESENT pLayer) see proportionally higher speedups.

---

## Correctness Verification

All implementations are verified through multiple strategies:

| Method | Description |
|--------|-------------|
| **CPU vs GPU byte-for-byte comparison** | GPU outputs matched against OpenMP and Numba JIT CPU outputs. |
| **Known-Answer Tests** | AES verified against NIST FIPS-197; GIFT, SKINNY, PRESENT against published test vectors. |
| **Cross-implementation consistency** | Single-threaded, OpenMP, Numba JIT, and CUDA outputs compared for identical inputs. |
| **Deterministic inputs** | Fixed keys and structured plaintext patterns ensure reproducibility. |
| **Inline correctness gate** | Any mismatch immediately aborts the benchmarking process. |

---

## Reproducing Results

### SPN Ciphers (AES, GIFT, SKINNY, PRESENT)

```bash
cd SPN_based

# Run verification and benchmarks for each cipher
python -m aes.verify && python -m aes.benchmark
python -m gift.verify && python -m gift.benchmark
python -m present.verify && python -m present.benchmark
python -m skinny.verify && python -m skinny.benchmark

# Generate comparison plots
python plot_mbps_graphs.py
```

### ARX Ciphers (SPECK, LEA, CHAM, Threefish)

```bash
cd ARX_based
pip install -r requirements.txt
python main.py
```

> **⚠️ CUDA Path Configuration:** Update the NVVM and CUDA driver paths in each `*_gpu.py` file to match your system:
> ```python
> os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
> os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"
> ```

### Feistel Ciphers (SIMECK, SIMON, TWINE, WARP)

Pre-compiled executables are included. Alternatively, recompile from source:

**Requirements:** CUDA 12.3, Visual Studio 2022, MSVC v143

```bash
cd Fiestel_based/SIMON

# Compile with optimizations
g++ -O3 -fopenmp -o simon simon.cpp           # CPU (OpenMP)
g++ -O3 -o simon_single simon_single.cpp      # CPU (single-threaded)
nvcc -O3 -o simongpu simon.cu                 # GPU (CUDA)

# Run
./simon.exe        # CPU OpenMP benchmark
./simongpu.exe     # GPU benchmark

# Generate plots
python simon.py
```

Repeat for SIMECK, TWINE, and WARP in their respective directories.

---

## Reports & Documentation

| Report | Location | Contents |
|--------|----------|---------|
| **Full Project Report** | [`GPU_Accelerated_Block_Ciphers_Report.pdf`](GPU_Accelerated_Block_Ciphers_Report.pdf) | Complete analysis across all 12 ciphers with figures and tables. |
| **ARX Report** | [`ARX_based/report.md`](ARX_based/report.md) | Detailed ARX cipher descriptions, GPU architecture, benchmark tables, and plots. |
| **Feistel Report** | [`Fiestel_based/report.tex`](Fiestel_based/report.tex) | LaTeX report covering CPU/GPU implementation, parallelization, and memory management. |
| **SPN Report** | [`SPN_based/SPN_CIPHER_OPTIMIZATIONS.md`](SPN_based/SPN_CIPHER_OPTIMIZATIONS.md) | Comprehensive SPN optimization strategies, S-box implementations, and full benchmark results. |