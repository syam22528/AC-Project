# SPN Cipher GPU Acceleration — Optimizations & Technical Reference

> **Project**: GPU-accelerated benchmarking of four Substitution-Permutation Network (SPN) ciphers  
> **Ciphers**: AES-128, GIFT-64-128, PRESENT-128, SKINNY-64-128  
> **Stack**: Python 3.10 · Numba JIT (CPU) · Numba CUDA JIT (GPU) · NumPy · Matplotlib  
> **Modes**: ECB (fully parallel) · CTR (counter-mode with XOR keystream)

---

## 1. Cipher Specifications

| Cipher | Block Size | Key Size | Rounds | State Representation |
|--------|-----------|----------|--------|---------------------|
| AES-128 | 128-bit (16 bytes) | 128-bit | 10 | Column-major byte array `uint8[16]` |
| GIFT-64-128 | 64-bit (8 bytes) | 128-bit | 28 | Packed `uint64` |
| SKINNY-64-128 | 64-bit (8 bytes) | 128-bit | 36 | Packed `uint64` (4×16-bit rows) |
| PRESENT-128 | 64-bit (8 bytes) | 128-bit | 31 + final | Packed `uint64` |

---

## 2. GPU Kernel Architecture

### 2.1 Thread Configuration

| Parameter | AES | GIFT | SKINNY | PRESENT |
|-----------|-----|------|--------|---------|
| Thread-block size | 256 | 256 | 256 | Occupancy-optimised (queried at construction) |
| Thread coarsening | **2 blocks/thread** | **4 blocks/thread** | **4 blocks/thread** | **8 blocks/thread** |
| Grid size | ⌈N / (256 × 2)⌉ | ⌈N / (256 × 4)⌉ | ⌈N / (256 × 4)⌉ | ⌈N / (block_size × 8)⌉ |

**Thread coarsening rationale**: Each CUDA thread processes multiple cipher blocks to amortise shared-memory setup overhead, improve arithmetic-to-bandwidth ratio, and hide global-memory latency. PRESENT uses the most aggressive coarsening (8×) because its delta-swap pLayer is computationally cheap in registers, making per-thread overhead more dominant.

### 2.2 S-Box Implementation Strategies

Each cipher offers multiple S-box strategies for GPU performance comparison:

#### AES-128

| Strategy | Description | Memory | Key Observation |
|----------|-------------|--------|-----------------|
| **Shared-memory S-box** (only variant) | 256-byte S-box loaded cooperatively into shared memory at kernel start | Shared (48 KB per SM) | High reuse: 16 lookups/round × 10 rounds = 160 accesses per block |

AES does **not** offer a bitsliced variant — a prototype was built but removed after benchmarking showed it consistently underperformed due to excessive register pressure (<0.5× throughput).

#### GIFT-64-128

| Strategy | Description | Key Technique |
|----------|-------------|---------------|
| **`table`** (default) | Fused SubNibbles + PermBits via a 256-entry constant-memory scatter table (`SP_SCATTER`) | Replaces 80 loop iterations (16 S-box + 64 P-box) with **16 table lookups** |
| **`bitsliced`** | GF(2) Boolean equations for SubNibbles; nibble scatter table for PermBits | All 16 nibbles processed simultaneously with zero memory-access latency |

The `table` variant precomputes `SP_SCATTER[k*16 + v] = pbox_scatter(k, SBOX[v])` at import time, fusing substitution and permutation into a single lookup per nibble.

#### SKINNY-64-128

| Strategy | Description | Key Technique |
|----------|-------------|---------------|
| **`table`** (default) | Dual-nibble 8-bit constant-memory table (`SBOX8`) | Processes two nibbles per byte in a single access, halving memory operations (8 lookups instead of 16) |
| **`bitsliced`** | Tableless Boolean equations with nibble-level inversion, AND, XOR, and rotation | Zero memory accesses beyond register file |

The `SBOX8` table is precomputed: `SBOX8[byte] = SBOX[byte & 0xF] | (SBOX[byte >> 4] << 4)`.

#### PRESENT-128

| Strategy | Description | Key Technique |
|----------|-------------|---------------|
| **`bitsliced`** (default) | GF(2) Boolean equations operating on all 16 nibbles simultaneously | Bit-planes extracted via `0x1111…` mask, processed in parallel, recombined |
| **`table`** | Standard 16-entry constant-memory S-box, one nibble at a time | Simpler implementation; useful for comparison |

### 2.3 Permutation / Linear Layer Optimizations

#### AES MixColumns — Branchless `xtime`

The `xtime` function (multiplication by 2 in GF(2⁸)) is implemented without branches:

```python
xtime(x) = ((x << 1) ^ (((x >> 7) & 1) * 0x1B)) & 0xFF
```

This avoids divergent warps on the GPU. The full MixColumns uses the compound-XOR trick, requiring only three `xtime` calls per column instead of full matrix multiplication.

#### AES ShiftRows

Implemented as explicit register-to-register byte swaps (no loops, no conditionals).

#### PRESENT pLayer — Delta-Swap Decomposition

The PRESENT permutation `P(i) = (16i) mod 63` (bit 63 fixed) is traditionally a 63-iteration bit loop. We decompose it into **4 delta-swap operations**:

```python
x = delta_swap(x, 0x0A0A0A0A0A0A0A0A, 3)   # swap index bits 0 ↔ 2
x = delta_swap(x, 0x0000F0F00000F0F0, 12)   # swap index bits 2 ↔ 4
x = delta_swap(x, 0x00CC00CC00CC00CC, 6)    # swap index bits 1 ↔ 3
x = delta_swap(x, 0x00000000FF00FF00, 24)   # swap index bits 3 ↔ 5
```

Each `delta_swap(x, mask, shift)` is 3 XOR + 1 AND + 1 shift = **5 instructions**. Total: **20 instructions** instead of ~189 (63 iterations × 3 ops). Correctness verified against the reference loop on 100K random test vectors.

#### GIFT PermBits — Nibble Scatter Table

Instead of a 64-iteration bit permutation loop, GIFT uses a precomputed 256-entry `PBOX_SCATTER` table:

```
PBOX_SCATTER[k*16 + v] = (bits of nibble value v scattered to output positions for nibble k)
```

This reduces PermBits to **16 table lookups + 16 OR operations**.

The **fused `SP_SCATTER`** table goes further — combining SubNibbles and PermBits into a single table, so one round needs only **16 lookups** total for both operations.

#### SKINNY Linear Layer — Row-Decomposed ShiftRows + MixColumns

The 64-bit SKINNY state is decomposed into four 16-bit rows:

```
row0 = bits [15:0]    (no shift)
row1 = bits [31:16]   (rotate right 4)
row2 = bits [47:32]   (rotate right 8)
row3 = bits [63:48]   (rotate right 12)
```

ShiftRows uses 16-bit rotations. MixColumns is a lightweight XOR-only mixing:

```
new_row0 = row3 ⊕ row1 ⊕ row2
new_row1 = row0
new_row2 = row1 ⊕ row2
new_row3 = row2 ⊕ row0
```

No GF(2⁸) multiplication needed (unlike AES), so the entire linear layer is pure XOR and rotate.

---

## 3. GPU Memory Management

### 3.1 Memory Placement Summary

| Data Item | Cipher(s) | GPU Memory Tier | Size | Justification |
|-----------|-----------|-----------------|------|---------------|
| Plaintext buffer | All | **Global** | ≤64 MB (AES @ 4M blocks) | Too large for shared/constant. Sequential coalesced access pattern. |
| Ciphertext buffer | All | **Global** | Same as plaintext | Same size and access pattern. |
| AES S-box (256 B) | AES | **Shared** | 256 bytes/block | 160 lookups per cipher block; shared memory eliminates repeated global loads. |
| 4-bit S-box (16 B) | GIFT, SKINNY, PRESENT | **Constant** | 16 bytes | Tiny read-only table; constant cache broadcasts to all warp threads. |
| GIFT P-box map | GIFT | **Constant** | 64 bytes | Fixed index array; uniform read-only broadcast access. |
| GIFT SP_SCATTER | GIFT | **Constant** | 2 KB (256 × uint64) | Fused S-box + P-box table; fits in constant cache. |
| SKINNY SBOX8 | SKINNY | **Constant** | 256 bytes | Dual-nibble packed table; uniform access. |
| Round keys (global mode) | AES | **Global** | 176 bytes | Flexible; supports arbitrary keys without kernel recompilation. |
| Round keys (constkeys mode) | AES | **Constant** | 176 bytes | Baked into a key-specialized kernel via `cuda.const.array_like`. |
| Round keys / masks | GIFT, SKINNY, PRESENT | **Shared** | 224–288 bytes | Staged once per thread block, reused across all rounds (28–36). |

### 3.2 Key Insight: Global vs Constant Memory for AES Round Keys

Two AES key-storage modes are benchmarked:

- **`global`**: Round keys stored in device global memory, passed as a kernel argument.
- **`constkeys`**: Round keys embedded in constant memory via a key-specialised kernel (compiled once per unique key, cached by key bytes).

**Finding**: Constant memory is *not* universally superior. At large batch sizes (4M blocks), AES with `global` achieves **1,011 MB/s** vs **927 MB/s** for `constkeys`. The 176-byte key schedule causes constant-cache thrashing when threads access round-dependent indices, making global memory's L2 cache more efficient for this access pattern.

### 3.3 PRESENT-Specific Memory Optimizations

PRESENT implements additional memory optimizations not present in the other ciphers:

- **Pinned host memory**: `cuda.pinned_array()` allocated once and reused across encryption calls, avoiding repeated page-locking overhead.
- **Persistent device buffers**: Device arrays allocated lazily and grown on demand, reused if the block count matches.
- **Occupancy-aware block size**: Queried at construction time via `cuda.occupancy.max_potential_block_size()` to maximise SM utilisation.
- **Native CTR kernels**: Counter blocks are generated and XORed with plaintext directly on the GPU in a single fused kernel pass, avoiding the separate ECB + host-side XOR path used by the other ciphers.

---

## 4. CPU Baseline Optimizations

All CPU implementations use Numba JIT compilation with the following optimizations:

| Optimization | Details |
|-------------|---------|
| **JIT compilation** | `@njit(cache=True)` — compiled once, cached to disk for subsequent runs |
| **Parallel blocks** | `prange` (OpenMP backend) distributes cipher blocks across CPU threads |
| **Configurable workers** | Thread count clamped to `os.cpu_count()` to avoid oversubscription |
| **Thread restore** | Worker count restored via `try/finally` to avoid side effects on the Numba thread pool |
| **Round-key caching** | Key schedule recomputed only when the key bytes change; cached result reused |
| **Zero-copy input** | `np.frombuffer()` avoids copying plaintext bytes into a new array |

Default benchmark configuration: **8 parallel workers** via `prange`.

---

## 5. Key Schedule Handling

| Cipher | Schedule Size | Computation | Placement |
|--------|--------------|-------------|-----------|
| AES-128 | 176 bytes (11 × 16 B) | Host-side: RotWord + SubWord + RCON expansion | Uploaded once per `set_key()` |
| GIFT-64-128 | 224 bytes (28 × uint64) | Host-side: 128-bit nibble register with rotation + LFSR update | Uploaded once per `set_key()` |
| SKINNY-64-128 | 144 bytes (36 × uint32) | Host-side: tweakey PT permutation + TK2 LFSR-2 update | Uploaded once per `set_key()` |
| PRESENT-128 | 256 bytes (32 × uint64) | Host-side: 128-bit register rotation + S-box + round counter XOR | Uploaded once per `set_key()` |

**AES constkeys caching**: The compiled key-specialized kernel is cached in a class-level dictionary keyed by `(variant, key_bytes)`. Reusing the same key skips recompilation entirely.

---

## 6. CTR Mode Implementation

CTR mode is implemented via a shared `ctr_utils` module used by all four ciphers:

### Counter Block Construction (`build_ctr_blocks`)

```
┌──────────────┬──────────────────┐
│    nonce     │    counter       │
│  (upper bytes) │  (lower bytes, big-endian) │
└──────────────┴──────────────────┘
```

- **Fast path** (counter ≤ 8 bytes): Vectorised NumPy construction with `np.arange` for counter values, avoiding per-block Python `int.to_bytes()` overhead.
- **Fallback**: Generic `bytearray` loop for unusual block sizes.

### CPU CTR Path

1. Generate counter blocks on the host (NumPy vectorised).
2. Encrypt counter blocks via ECB.
3. XOR keystream with plaintext via `np.bitwise_xor()`.

### GPU CTR Path (generic)

Same as CPU CTR, but step 2 runs the GPU ECB kernel. The XOR is performed host-side after device→host copy.

### GPU Native CTR Path (PRESENT only)

PRESENT implements dedicated on-device CTR kernels (`present_encrypt_ctr_kernel_bitsliced` / `_table`) that:

1. Compute `counter_value = base_ctr_block + block_index` directly on the GPU.
2. Encrypt the counter value through 31 rounds.
3. XOR the encrypted counter with the plaintext block — all in one kernel launch.

This eliminates the separate ECB pass and host-side XOR, reducing memory traffic and kernel launch overhead.

---

## 7. Benchmarking Methodology

### 7.1 CPU/GPU Comparison Design

```
For each block_count:
    1. Warm up CPU JIT (1 call, result discarded)
    2. Measure CPU: median of N runs
    3. For each GPU variant:
        a. Warm up GPU kernel (2 calls, results discarded)
        b. Measure GPU: median of N runs
        c. Record total / kernel / transfer times separately
        d. Assert CPU_ciphertext == GPU_ciphertext (correctness gate)
```

Key design choices:
- **CPU timed once per block size** — all GPU variants and key modes share the same CPU baseline, avoiding redundant CPU work.
- **Correctness gate** — CPU/GPU output is compared byte-for-byte at every data point. Any mismatch aborts the benchmark immediately.
- **Median of 3 runs** (default) — robust to outliers from OS jitter.

### 7.2 Timing Breakdown

| Metric | How Measured | What It Captures |
|--------|-------------|-----------------|
| `cpu_seconds` | `time.perf_counter()` around CPU encrypt call | Full CPU encryption time |
| `gpu_total_seconds` | `time.perf_counter()` around entire GPU pipeline | H2D + kernel + D2H |
| `gpu_kernel_seconds` | `cuda.event_elapsed_time()` (AES, GIFT, SKINNY) or `perf_counter` around `cuda.synchronize()` (PRESENT) | Pure kernel execution |
| `gpu_transfer_seconds` | `gpu_total - gpu_kernel` or explicit `perf_counter` intervals | PCIe data movement overhead |

### 7.3 Transfer Overhead Analysis

At the largest batch size (4M blocks, ECB mode), PCIe transfer dominates GPU time for fast kernels:

| Cipher | Transfer % of GPU Total | Kernel-only Speedup | End-to-End Speedup |
|--------|------------------------|--------------------|--------------------|
| AES-128 | **89%** | ~40× | 4.4× |
| SKINNY-64-128 | 76% | — | 5.1× |
| GIFT-64-128 | 34% | — | 4.8× |
| PRESENT-128 | **16%** | — | 15.6× |

**Key insight**: AES has the highest absolute throughput (1,011 MB/s) but the lowest speedup (4.4×) because its CPU baseline is already fast (~230 MB/s). PRESENT achieves the highest speedup (15.6×) because its CPU pLayer implementation is very slow (~7 MB/s), making transfer overhead negligible.

---

## 8. Correctness Verification

Each cipher has a dedicated `verify.py` script that runs two levels of testing:

| Test Level | AES | GIFT | SKINNY | PRESENT |
|-----------|-----|------|--------|---------|
| **Known-answer vectors** | NIST FIPS-197 Appendix B | 3 published test vectors (all-zero, reference, mixed) | skinny-c reference vector | Published 128-bit key vector |
| **CPU/GPU cross-check** | 8,192 random blocks, both key modes (global + constkeys) | 32,768 random blocks, both variants (table + bitsliced) | 32,768 random blocks, both variants | 32,768 random blocks, both variants |

All benchmark scripts additionally perform inline CPU/GPU ciphertext comparison at every data point.

---

## 9. Performance Results Summary (4M Blocks, ECB, Best Variant)

| Cipher | Best Variant | CPU Time (s) | GPU Total (s) | Kernel (s) | Transfer (s) | GPU MB/s | Speedup |
|--------|-------------|-------------|---------------|-----------|-------------|---------|---------|
| AES-128 | global | 0.294 | 0.066 | 0.007 | 0.059 | **1,011** | 4.4× |
| AES-128 | constkeys | 0.290 | 0.072 | 0.011 | 0.061 | 927 | 4.0× |
| GIFT-64-128 | table | 0.644 | 0.134 | 0.089 | 0.045 | 251 | 4.8× |
| SKINNY-64-128 | bitsliced | 0.521 | 0.102 | 0.024 | 0.078 | 329 | 5.1× |
| PRESENT-128 | bitsliced | 3.937 | 0.252 | 0.213 | 0.039 | 133 | **15.6×** |

- **Highest throughput**: AES-128 (global) — 1,011 MB/s
- **Highest speedup**: PRESENT-128 (bitsliced) — 15.6×
- **CTR mode**: Typically 1.5–2× lower throughput than ECB due to counter construction and XOR overhead

---

## 10. Project File Structure

```
SNP_Ciphers/
├── ctr_utils.py              # Shared CTR-mode utilities (counter block generation + XOR)
├── plot_mbps_graphs.py        # Matplotlib visualisation: 5 benchmark figures
├── aes/
│   ├── __init__.py            # Package exports
│   ├── cpu.py                 # CPU AES-128: Numba JIT, ECB + CTR
│   ├── gpu.py                 # GPU AES-128: shared-memory S-box, global/constkeys round keys
│   ├── benchmark.py           # CPU vs GPU benchmark driver (CSV + table output)
│   └── verify.py              # NIST vector + CPU/GPU consistency checks
├── gift/
│   ├── __init__.py
│   ├── common.py              # S-box, P-box, round constants, key schedule
│   ├── cpu.py                 # CPU GIFT-64-128: table S-box, Numba parallel
│   ├── gpu.py                 # GPU GIFT-64-128: table (SP_SCATTER) + bitsliced variants
│   ├── benchmark.py
│   └── verify.py
├── present/
│   ├── __init__.py
│   ├── common.py              # S-box, P-box, key schedule (128-bit rotation + LFSR)
│   ├── cpu.py                 # CPU PRESENT-128: delta-swap pLayer, Numba parallel
│   ├── gpu.py                 # GPU PRESENT-128: bitsliced + table, pinned memory, native CTR
│   ├── benchmark.py
│   └── verify.py
├── skinny/
│   ├── __init__.py
│   ├── common.py              # S-box, tweakey schedule (PT permutation + TK2 LFSR-2)
│   ├── cpu.py                 # CPU SKINNY-64-128: table S-box, row decomposition
│   ├── gpu.py                 # GPU SKINNY-64-128: dual-nibble SBOX8 table + bitsliced
│   ├── benchmark.py
│   └── verify.py
└── results/
    ├── aes_benchmark.csv
    ├── gift_benchmark.csv
    ├── present_benchmark.csv
    ├── skinny_benchmark.csv
    └── plots/
        ├── throughput_ecb.png
        ├── throughput_ctr.png
        ├── speedup_ecb.png
        ├── speedup_ctr.png
        └── cipher_comparison.png
```

---

## 11. Optimization Strategy Summary

### GPU Kernel Optimizations

| Technique | Where Used | Effect |
|-----------|-----------|--------|
| Thread coarsening (2–8 blocks/thread) | All ciphers | Amortises shared-memory setup overhead |
| Shared-memory S-box (256 B) | AES | Eliminates 160 global-memory lookups per block |
| Shared-memory round-key staging | GIFT, SKINNY, PRESENT | Single load per thread-block, reused 28–36 times |
| Constant-memory S-box/P-box | GIFT, SKINNY, PRESENT | Cached broadcast to all warp threads |
| Fused SP scatter table | GIFT (table variant) | Combines SubNibbles + PermBits into 16 lookups |
| Dual-nibble SBOX8 | SKINNY (table variant) | Halves S-box memory operations (8 vs 16) |
| Bitsliced S-box (Boolean logic) | GIFT, SKINNY, PRESENT | Zero memory latency; all nibbles processed in parallel |
| Delta-swap pLayer | PRESENT | 4 operations replace 63-iteration bit loop |
| Branchless `xtime` MixColumns | AES | Eliminates warp divergence |
| Packed `uint64` state | GIFT, SKINNY, PRESENT | Single register per cipher block; enables bitwise parallelism |
| Pinned host memory | PRESENT | Avoids repeated page-locking overhead |
| Occupancy-aware block size | PRESENT | Maximises SM utilisation via CUDA occupancy API |
| Native CTR kernel | PRESENT | Fuses counter generation + encryption + XOR in one kernel |
| Constant-memory round keys | AES (constkeys) | Key-specialized kernel; trades flexibility for cache locality |

### CPU Baseline Optimizations

| Technique | Effect |
|-----------|--------|
| `@njit(cache=True)` | Compiled once, cached to disk |
| `prange` (OpenMP) | Parallel block processing across 8 CPU threads |
| Round-key caching | Avoids recomputation on repeated encrypt calls with same key |
| Thread count clamping | Prevents oversubscription beyond logical core count |

### Infrastructure Optimizations

| Technique | Effect |
|-----------|--------|
| Shared CPU baseline per block size | Avoids redundant CPU timing when evaluating multiple GPU variants |
| Inline correctness gate | Catches CPU/GPU divergence immediately during benchmarking |
| Warm-up runs before timing | Excludes JIT compilation overhead from measurements |
| Median of 3 runs | Robust to OS scheduling jitter |
| Vectorised NumPy counter blocks | Fast CTR-mode counter construction for large batch sizes |
