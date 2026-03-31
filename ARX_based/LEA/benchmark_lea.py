import time
import numpy as np
from numba import cuda

from LEA.lea_naive import encrypt as naive_enc
from LEA.lea_optimized import encrypt as opt_enc
from LEA.lea_numba import encrypt as numba_enc

from LEA.lea_gpu import lea_kernel, expand_key, bytes_to_blocks


# -----------------------------
# GPU detailed timing
# -----------------------------

def gpu_benchmark_detailed(data, key):
    blocks = bytes_to_blocks(data)
    round_keys = expand_key(key)

    threads_per_block = 256
    blocks_per_grid = (blocks.shape[0] + threads_per_block - 1) // threads_per_block

    # -----------------------------
    # Transfer: Host → Device
    # -----------------------------
    t0 = time.perf_counter()
    d_blocks = cuda.to_device(blocks)
    d_out = cuda.device_array_like(blocks)
    d_keys = cuda.to_device(round_keys)
    t1 = time.perf_counter()

    h2d_time = t1 - t0

    # -----------------------------
    # Kernel timing (CUDA events)
    # -----------------------------
    start_event = cuda.event()
    end_event = cuda.event()

    start_event.record()

    lea_kernel[blocks_per_grid, threads_per_block](
        d_blocks, d_out, d_keys
    )

    end_event.record()
    end_event.synchronize()

    kernel_time = cuda.event_elapsed_time(start_event, end_event) / 1000.0  # sec

    # -----------------------------
    # Transfer: Device → Host
    # -----------------------------
    t2 = time.perf_counter()
    result = d_out.copy_to_host()
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


def benchmark_lea():
    key = bytes(range(16))
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
 
        if size <= 1024 * 1024:
            t_naive = cpu_benchmark(naive_enc, data, key)
            t_opt   = cpu_benchmark(opt_enc,   data, key)
            print(f"Naive Time       : {t_naive:.6f}s")
            print(f"Optimized Time   : {t_opt:.6f}s")
            row["t_naive"] = t_naive
            row["t_opt"]   = t_opt
        else:
            print("Naive/Optimized  : Skipped")
            row["t_naive"] = None
            row["t_opt"]   = None
 
        numba_enc(data[:64], key)  # warm-up
        t_numba = cpu_benchmark(numba_enc, data, key)
        print(f"Numba Time       : {t_numba:.6f}s")
        row["t_numba"] = t_numba
 
        total_gpu, kernel_gpu, mem_gpu = gpu_benchmark_detailed(data, key)
        print(f"GPU Total Time   : {total_gpu:.6f}s")
        print(f"GPU Kernel Time  : {kernel_gpu:.6f}s")
        print(f"GPU Mem Time     : {mem_gpu:.6f}s")
        row["t_gpu_total"]  = total_gpu
        row["t_gpu_kernel"] = kernel_gpu
        row["t_gpu_mem"]    = mem_gpu
 
        gpu_throughput   = mb / total_gpu
        numba_throughput = mb / t_numba
        blocks           = size // 16
        gpu_blocks_per_sec = blocks / total_gpu
        speedup          = t_numba / total_gpu
 
        print(f"GPU Throughput   : {gpu_throughput:.2f} MB/s")
        print(f"Numba Throughput : {numba_throughput:.2f} MB/s")
        print(f"GPU Blocks/sec   : {gpu_blocks_per_sec:.2f}")
        print(f"Speedup (Numba/GPU): {speedup:.2f}x")
 
        row["gpu_throughput"]    = gpu_throughput
        row["numba_throughput"]  = numba_throughput
        row["gpu_blocks_per_sec"] = gpu_blocks_per_sec
        row["speedup"]           = speedup
 
        results.append(row)
 
    return results

if __name__ == "__main__":
    benchmark_lea()