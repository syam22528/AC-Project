# ⚡ AC-Project — GPU-Accelerated Lightweight Cipher Benchmarking

> **Systematic performance evaluation of 12 block ciphers across three cryptographic families, benchmarked on CPU and GPU (CUDA) with multiple optimization tiers.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Cipher Families](#cipher-families)
  - [ARX-Based Ciphers](#1-arx-based-ciphers)
  - [Feistel-Based Ciphers](#2-feistel-based-ciphers)
  - [SPN-Based Ciphers](#3-spn-based-ciphers)
- [Project Structure](#project-structure)
- [Hardware & Software Setup](#hardware--software-setup)
- [Installation & Usage](#installation--usage)
- [Implementation Details](#implementation-details)
  - [ARX Implementation Tiers](#arx-implementation-tiers)
  - [Feistel Implementation Tiers](#feistel-implementation-tiers)
  - [SPN Implementation Tiers](#spn-implementation-tiers)
- [GPU Optimization Techniques](#gpu-optimization-techniques)
- [Benchmark Results Summary](#benchmark-results-summary)
- [Correctness Verification](#correctness-verification)
- [Reports & Documentation](#reports--documentation)
- [License](#license)

---

## Overview

This project provides a comprehensive benchmarking framework for **12 lightweight and standard block ciphers** organized into three cryptographic families:

| Family | Ciphers | Language | GPU Backend |
|--------|---------|----------|-------------|
| **ARX** (Add–Rotate–XOR) | SPECK, LEA, CHAM, Threefish | Python | Numba CUDA |
| **Feistel** | SIMON, SIMECK, TWINE, WARP | C++ | NVIDIA CUDA C |
| **SPN** (Substitution–Permutation Network) | AES, GIFT, PRESENT, SKINNY | Python | Numba CUDA |

Each cipher is implemented with **multiple optimization tiers** — from naïve reference implementations to GPU-accelerated CUDA kernels — and benchmarked across a wide range of input sizes (1 KB – 1.6 GB) to measure:

- **Throughput** (MB/s or GB/s)
- **Speedup** (GPU vs. CPU baseline)
- **Kernel execution time** vs. **memory transfer overhead**
- **Mode-of-operation impact** (ECB and CTR)

---

## Cipher Families

### 1. ARX-Based Ciphers

Located in [`ARX_based/`](ARX_based/)

| Cipher | Block Size | Key Size | Rounds | Structure |
|--------|-----------|----------|--------|-----------|
| **SPECK 128/128** | 128-bit | 128-bit | 32 | ARX (Feistel-like) |
| **LEA-128** | 128-bit | 128-bit | 24 | ARX (Generalized Feistel Network) |
| **CHAM-128/128** | 128-bit | 128-bit | 80 | ARX (Generalized Feistel Network) |
| **Threefish-256** | 256-bit | 256-bit | 72 | ARX (Substitution–Permutation) |

**Key highlights:**
- All four implementations share a consistent Python + Numba CUDA architecture
- GPU kernels use **shared memory** for round keys and **grid-stride loops** for arbitrary input sizes
- Peak GPU throughput converges to **~550 MB/s** across all ciphers, revealing PCIe transfer as the bottleneck
- Best speedup: **119× (SPECK at 10 MB)**, **93.6× sustained (LEA at 100 MB)**

---

### 2. Feistel-Based Ciphers

Located in [`Fiestel_based/`](Fiestel_based/)

| Cipher | Block Size | Key Size | Rounds | Structure |
|--------|-----------|----------|--------|-----------|
| **SIMON 64/128** | 64-bit | 128-bit | 44 | Feistel (bitwise rotation, AND, XOR) |
| **SIMECK 64/128** | 64-bit | 128-bit | 44 | Feistel (simplified SIMON variant) |
| **TWINE-80** | 64-bit | 80-bit | 36 | Generalized Feistel (4-bit S-box) |
| **WARP** | 128-bit | 128-bit | 41 | Generalized Feistel (4-bit S-box) |

**Key highlights:**
- Implemented in **C++ (OpenMP)** for CPU and **CUDA C** for GPU
- **Bit-slicing optimization** used for TWINE and WARP (32 blocks processed simultaneously)
- CUDA kernels use **constant memory** for round keys and permutation tables
- Visualization scripts (Python/Matplotlib) for generating performance comparison plots

---

### 3. SPN-Based Ciphers

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
- **Peak GPU throughput: 1,346 MB/s (AES)**; **Peak speedup: 14.21× (PRESENT CTR)**

---

## Project Structure

```
AC-Project/
│
├── README.md                          ← This file
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
│   │   ├── time_vs_size.png
│   │   ├── speedup.png
│   │   ├── throughput.png
│   │   ├── gpu_breakdown.png
│   │   └── bar_100mb.png
│   ├── main.py                        # Entry point: benchmarks + plot generation
│   ├── output.txt                     # Sample benchmark output
│   ├── requirements.txt               # Python dependencies
│   └── report.md                      # Detailed ARX analysis report
│
├── Fiestel_based/                     ← Feistel cipher implementations (C++ / CUDA C)
│   ├── SIMON/
│   │   ├── simon.cpp                  # CPU (OpenMP multi-threaded)
│   │   ├── simon_single.cpp           # CPU (single-threaded baseline)
│   │   ├── simon.cu                   # CUDA GPU kernel
│   │   ├── simon.py                   # Plotting / visualization script
│   │   └── simon_three_columns_updated.png
│   ├── SIMECK/                        # Same structure as SIMON
│   ├── TWINE/                         # Same structure (+ bit-slicing)
│   ├── WARP/                          # Same structure (+ bit-slicing)
│   └── report.tex                     # LaTeX report for Feistel ciphers
│
├── SPN_based/                         ← SPN cipher implementations (Python + Numba CUDA)
│   ├── aes/
│   │   ├── __init__.py
│   │   ├── cpu.py                     # Numba JIT AES-128 (ECB + CTR)
│   │   ├── gpu.py                     # GPU: shared S-box, global/constkeys modes
│   │   ├── benchmark.py               # Benchmark driver
│   │   └── verify.py                  # NIST test vector verification
│   ├── gift/                          # Table + bitsliced GPU variants
│   ├── present/                       # Bitsliced + table variants, native CTR kernel
│   ├── skinny/                        # Dual-nibble SBOX8 + bitsliced variants
│   ├── results/
│   │   ├── aes_benchmark.csv
│   │   ├── gift_benchmark.csv
│   │   ├── present_benchmark.csv
│   │   ├── skinny_benchmark.csv
│   │   └── plots/                     # Generated benchmark plots (PNG)
│   ├── ctr_utils.py                   # Shared CTR-mode utilities
│   ├── plot_mbps_graphs.py            # Plot generation script
│   └── SPN_CIPHER_OPTIMIZATIONS.md    # Detailed SPN optimization report
```

---

## Hardware & Software Setup

The benchmarks were run on the following configurations:

### ARX & SPN Ciphers

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Ultra 7 155H |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop (65W TDP, 8 GB) |
| **RAM** | 16 GB |
| **OS** | Windows 11 |
| **CUDA Toolkit** | 11.8 |
| **Python** | 3.10+ |
| **GPU Backend** | Numba CUDA JIT |

### Feistel Ciphers

| Component | Specification |
|-----------|--------------|
| **CPU** | AMD Ryzen 7 5800H (8 cores / 16 threads) |
| **GPU** | NVIDIA RTX 3050 Ti (95W TDP) |
| **RAM** | 16 GB |
| **OS** | Windows 11 |
| **CUDA Toolkit** | 12.3 |
| **Compiler** | Visual Studio 2022 (MSVC v143) |

---

## Installation & Usage

### Prerequisites

- **Python 3.8+** (for ARX and SPN ciphers)
- **NVIDIA GPU** with CUDA-compatible drivers
- **CUDA Toolkit** (v11.8 or v12.x)
- **Visual Studio 2022** (for Feistel C++/CUDA compilation)

### ARX Ciphers

```bash
cd ARX_based
pip install -r requirements.txt
```

> **⚠️ CUDA Path Configuration:** Update the NVVM and CUDA driver paths in each `*_gpu.py` file to match your system:
> ```python
> os.environ["NUMBA_CUDA_NVVM"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\nvvm\bin\nvvm.dll"
> os.environ["NUMBA_CUDA_DRIVER"] = r"C:\Windows\System32\nvcuda.dll"
> ```

**Run benchmarks:**
```bash
python main.py
```

**Run correctness tests** (uncomment in `main.py`):
```python
test_speck_consistency()
test_lea_consistency()
test_cham_consistency()
test_threefish_consistency()
```

### Feistel Ciphers

Each cipher has three compilation targets:

```bash
cd Fiestel_based/SIMON

# Single-threaded CPU baseline
g++ -O2 -o simon_single simon_single.cpp

# Multi-threaded CPU (OpenMP)
g++ -O2 -fopenmp -o simon simon.cpp

# GPU (CUDA)
nvcc -O2 -o simongpu simon.cu
```

Or with MSVC (Visual Studio):
```bash
cl /O2 /openmp simon.cpp        # CPU OpenMP
nvcc -O2 simon.cu -o simongpu   # GPU CUDA
```

**Generate plots:**
```bash
python simon.py   # Generates simon_three_columns_updated.png
```

### SPN Ciphers

```bash
cd SPN_based

# Run individual benchmarks
python -m aes.benchmark
python -m gift.benchmark
python -m present.benchmark
python -m skinny.benchmark

# Generate comparison plots
python plot_mbps_graphs.py
```

---

## Implementation Details

### ARX Implementation Tiers

Each ARX cipher is implemented in four progressive optimization tiers:

| Tier | File Suffix | Description |
|------|-------------|-------------|
| **Naive** | `_naive.py` | Pure Python, block-by-block loop. Correctness reference. |
| **Optimized** | `_optimized.py` | Pure Python with batch processing and structural improvements. |
| **Numba JIT** | `_numba.py` | NumPy arrays + `@njit` compiled loops. Single-threaded native code via LLVM. |
| **CUDA GPU** | `_gpu.py` | Numba CUDA kernels with shared memory for round keys and grid-stride loops. |

### Feistel Implementation Tiers

| Tier | File | Description |
|------|------|-------------|
| **Single-threaded** | `*_single.cpp` | Sequential C++ baseline for correctness and reference timing. |
| **Multi-threaded** | `*.cpp` | OpenMP-parallelized across plaintext blocks. |
| **GPU** | `*.cu` | CUDA kernels with constant memory for keys and grid-stride loops. |

TWINE and WARP additionally use **bit-slicing** (32 blocks processed simultaneously per thread) on both CPU and GPU.

### SPN Implementation Tiers

| Tier | Description |
|------|-------------|
| **CPU (Numba JIT)** | `@njit(cache=True)` with `prange` for multi-core parallel block processing. |
| **GPU (Table variant)** | Lookup-table-based S-box in shared/constant memory. |
| **GPU (Bitsliced variant)** | GF(2) Boolean equations — zero memory-access latency. |
| **GPU (Constkeys)** | AES-specific: round keys baked into constant memory via specialized kernel. |

---

## GPU Optimization Techniques

### Shared Across All Families

| Technique | Description |
|-----------|-------------|
| **Grid-stride loops** | Each thread processes multiple blocks; handles arbitrary input sizes. |
| **Shared/Constant memory** | Round keys and S-boxes placed in fast on-chip memory. |
| **256 threads/block** | Aligned with warp scheduling for optimal GPU occupancy. |

### ARX-Specific

| Technique | Impact |
|-----------|--------|
| Shared memory round keys | Eliminates repeated global memory fetches per round. |
| CUDA event timing | Precise kernel vs. transfer time decomposition. |

### Feistel-Specific

| Technique | Impact |
|-----------|--------|
| Constant memory (round keys, permutation tables) | Fast broadcast to all threads in a warp. |
| Bit-slicing (TWINE, WARP) | 32 blocks per thread, SIMD-style parallelism. |
| `#pragma GCC unroll` | Loop unrolling for SIMECK round function. |

### SPN-Specific

| Technique | Cipher | Impact |
|-----------|--------|--------|
| Thread coarsening (2–8 blocks/thread) | All | Amortizes shared-memory setup overhead. |
| Fused SP scatter table | GIFT | Replaces 80 loop iterations with 16 lookups. |
| Delta-swap pLayer decomposition | PRESENT | 4 operations replace a 63-iteration bit loop. |
| Dual-nibble SBOX8 | SKINNY | Halves S-box memory operations (8 vs 16 lookups). |
| Bitsliced Boolean S-box | GIFT, SKINNY, PRESENT | Zero memory latency; 4–12× faster kernel than table. |
| Branchless `xtime` MixColumns | AES | Eliminates warp divergence. |
| Pinned host memory | PRESENT | Avoids repeated page-locking overhead. |
| Native CTR kernel | PRESENT | Fuses counter + encrypt + XOR in one pass; **14.2× speedup**. |
| Occupancy-aware block sizing | PRESENT | Maximizes SM utilization via CUDA occupancy API. |

---

## Benchmark Results Summary

### ARX Ciphers — GPU Throughput @ 100 MB

| Cipher | GPU Throughput | Numba Throughput | Speedup |
|--------|---------------|-----------------|---------|
| **SPECK** | 571.67 MB/s | 15.49 MB/s | 36.91× |
| **LEA** | 581.65 MB/s | 6.21 MB/s | 93.61× |
| **CHAM** | 563.10 MB/s | 196.16 MB/s | 2.87× |
| **Threefish** | 530.83 MB/s | 178.73 MB/s | 2.97× |

> All ARX ciphers converge to ~530–580 MB/s GPU throughput, confirming **PCIe memory transfer** as the bottleneck.

### SPN Ciphers — Peak GPU Performance (ECB)

| Cipher | Best Variant | Peak Throughput | Peak Speedup |
|--------|-------------|----------------|-------------|
| **AES-128** | constkeys | **1,346 MB/s** | 9.90× |
| **GIFT-64-128** | table | 328 MB/s | 6.12× |
| **SKINNY-64-128** | bitsliced | 972 MB/s | 11.22× |
| **PRESENT-128** | bitsliced | 795 MB/s | 11.29× |

> **PRESENT achieves the highest CTR speedup** at **14.21×** thanks to its native on-device CTR kernel.

### Cross-Family Peak Performance

| Metric | Winner | Value |
|--------|--------|-------|
| Highest absolute throughput | AES (constkeys, ECB) | **1,346 MB/s** |
| Highest ECB speedup (SPN) | PRESENT (bitsliced) | **11.29×** |
| Highest CTR speedup | PRESENT (native CTR) | **14.21×** |
| Highest kernel-only speedup | AES (constkeys) | **170.9×** |
| Highest ARX speedup | SPECK (10 MB) | **119×** |
| Highest sustained ARX speedup | LEA (100 MB) | **93.6×** |

---

## Correctness Verification

All implementations are verified through multiple strategies:

| Method | Description |
|--------|-------------|
| **Encrypt–Decrypt Round-Trip** | `decrypt(encrypt(plaintext)) == plaintext` for all tiers. |
| **Cross-Implementation Consistency** | Naive, Optimized, Numba, and GPU outputs compared byte-for-byte. |
| **Known-Answer Tests** | AES verified against NIST FIPS-197 Appendix B; GIFT, SKINNY, and PRESENT against published test vectors. |
| **Inline Correctness Gate** | SPN benchmarks abort immediately on any CPU/GPU ciphertext mismatch. |
| **Deterministic Inputs** | Fixed keys and structured plaintext patterns ensure reproducibility. |

---

## Reports & Documentation

Each cipher family has its own detailed report:

| Report | Location | Format | Contents |
|--------|----------|--------|----------|
| **ARX Report** | [`ARX_based/report.md`](ARX_based/report.md) | Markdown | Cipher descriptions, GPU architecture, full benchmark tables, plots, and analysis. |
| **Feistel Report** | [`Fiestel_based/report.tex`](Fiestel_based/report.tex) | LaTeX | CPU/GPU implementation details, parallelization strategies, memory management. |
| **SPN Report** | [`SPN_based/SPN_CIPHER_OPTIMIZATIONS.md`](SPN_based/SPN_CIPHER_OPTIMIZATIONS.md) | Markdown | Optimization strategies, S-box implementations, full benchmark tables with trend analysis. |

---

## License

This project was developed as part of an **Applied Cryptography** course. For academic use only.