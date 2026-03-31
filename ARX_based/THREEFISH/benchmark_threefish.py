import time
import numpy as np
from numba import cuda

from THREEFISH.threefish_naive import encrypt as naive_enc
from THREEFISH.threefish_optimized import encrypt as opt_enc
from THREEFISH.threefish_numba import encrypt as numba_enc
from THREEFISH.threefish_gpu import encrypt as gpu_enc

from THREEFISH.threefish_gpu import threefish_encrypt_kernel


# -----------------------------
# Helper: bytes → uint64 blocks
# -----------------------------
def bytes_to_blocks(data):
    return np.frombuffer(data, dtype=np.uint64).reshape(-1, 4).copy()


# -----------------------------
# GPU detailed timing
# -----------------------------
def gpu_benchmark_detailed(data, key):

    # Key schedule (uint64)
    k = np.frombuffer(key, dtype=np.uint64).copy()
    C240 = np.uint64(0x1BD11BDAA9FC1A22)
    k4 = C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3]
    k = np.append(k, k4)

    # Padding (32 bytes!)
    pad_len = 32 - (len(data) % 32)
    data = data + bytes([pad_len]) * pad_len

    blocks = bytes_to_blocks(data)

    threads_per_block = 256
    blocks_per_grid = (blocks.shape[0] + threads_per_block - 1) // threads_per_block

    # -----------------------------
    # Transfer: Host → Device
    # -----------------------------
    t0 = time.perf_counter()
    d_blocks = cuda.to_device(blocks)
    d_k = cuda.to_device(k)
    t1 = time.perf_counter()

    h2d_time = t1 - t0

    # -----------------------------
    # Kernel timing
    # -----------------------------
    start_event = cuda.event()
    end_event = cuda.event()

    start_event.record()

    threefish_encrypt_kernel[blocks_per_grid, threads_per_block](d_blocks, d_k)

    end_event.record()
    end_event.synchronize()

    kernel_time = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    # -----------------------------
    # Transfer: Device → Host
    # -----------------------------
    t2 = time.perf_counter()
    result = d_blocks.copy_to_host()
    t3 = time.perf_counter()

    d2h_time = t3 - t2

    total_time = h2d_time + kernel_time + d2h_time

    return total_time, kernel_time, (h2d_time + d2h_time)


# -----------------------------
# CPU benchmark
# -----------------------------
def cpu_benchmark(func, data, key):
    start = time.perf_counter()
    func(data, key)
    end = time.perf_counter()
    return end - start


# -----------------------------
# Main benchmark
# -----------------------------
def benchmark_threefish():
    key = bytes(range(32))  #  FIXED

    # sizes = [
    #     1024,
    #     1024 * 1024,
    #     10 * 1024 * 1024,
    #     50 * 1024 * 1024,
    #     100 * 1024 * 1024,
    # ]
    sizes = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800 , 104857600]
    results = []

    for size in sizes:
        mb = size / (1024 * 1024)
        print(f"\n===== Size: {mb:.2f} MB =====")

        data = b'A' * size
        row = {"size_mb": mb}

        # -----------------------------
        # Naive + Optimized
        # -----------------------------
        if size <= 1024 * 1024:
            t_naive = cpu_benchmark(naive_enc, data, key)
            t_opt   = cpu_benchmark(opt_enc, data, key)

            print(f"Naive Time       : {t_naive:.6f}s")
            print(f"Optimized Time   : {t_opt:.6f}s")

            row["t_naive"] = t_naive
            row["t_opt"]   = t_opt
        else:
            print("Naive/Optimized  : Skipped")
            row["t_naive"] = None
            row["t_opt"]   = None

        # -----------------------------
        # Numba CPU
        # -----------------------------
        numba_enc(data[:64], key)  # warm-up

        t_numba = cpu_benchmark(numba_enc, data, key)
        print(f"Numba Time       : {t_numba:.6f}s")

        row["t_numba"] = t_numba

        # -----------------------------
        # GPU
        # -----------------------------
        total_gpu, kernel_gpu, mem_gpu = gpu_benchmark_detailed(data, key)

        print(f"GPU Total Time   : {total_gpu:.6f}s")
        print(f"GPU Kernel Time  : {kernel_gpu:.6f}s")
        print(f"GPU Mem Time     : {mem_gpu:.6f}s")

        row["t_gpu_total"]  = total_gpu
        row["t_gpu_kernel"] = kernel_gpu
        row["t_gpu_mem"]    = mem_gpu

        # -----------------------------
        # Metrics
        # -----------------------------
        gpu_throughput   = mb / total_gpu
        numba_throughput = mb / t_numba

        blocks = size // 32  #  FIXED
        gpu_blocks_per_sec = blocks / total_gpu

        speedup = t_numba / total_gpu

        print(f"GPU Throughput   : {gpu_throughput:.2f} MB/s")
        print(f"Numba Throughput : {numba_throughput:.2f} MB/s")
        print(f"GPU Blocks/sec   : {gpu_blocks_per_sec:.2f}")
        print(f"Speedup (Numba/GPU): {speedup:.2f}x")

        row["gpu_throughput"]     = gpu_throughput
        row["numba_throughput"]   = numba_throughput
        row["gpu_blocks_per_sec"] = gpu_blocks_per_sec
        row["speedup"]            = speedup

        results.append(row)

    return results


if __name__ == "__main__":
    benchmark_threefish()