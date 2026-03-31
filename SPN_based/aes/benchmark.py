from __future__ import annotations

"""AES-128 CPU vs GPU benchmark in ECB and CTR modes.

Measures throughput (MB/s) and GPU speedup for:
  - ECB: direct block cipher encryption — the baseline mode.
  - CTR: counter mode, implemented as ECB(counter) + XOR with plaintext.

The CPU is timed once per block size; all GPU key modes (global and constkeys)
are then compared against that same CPU baseline to avoid redundant timing.

Results are written to a CSV file and also printed as a formatted table.
"""

import argparse
import csv
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from aes.cpu import AesCpuOptimized
    from aes.gpu import AesGpuOptimized, has_cuda_gpu
else:
    from .cpu import AesCpuOptimized
    from .gpu import AesGpuOptimized, has_cuda_gpu


@dataclass
class BenchRow:
    """One row of benchmark results for a single (block_count, mode, key_mode) point."""
    mode: str
    gpu_variant: str
    key_mode: str
    cpu_workers: int
    cpu_impl: str
    blocks: int
    bytes_total: int
    cpu_seconds: float
    cpu_mbps: float
    gpu_total_seconds: float
    gpu_kernel_seconds: float
    gpu_transfer_seconds: float
    gpu_total_mbps: float
    gpu_kernel_mbps: float
    speedup_total: float
    speedup_kernel_only: float


def parse_block_sizes(text: str) -> list[int]:
    """Parse a comma-separated string of block counts into a list of integers."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_list(text: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated string of integers into a list of ints."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def throughput_mbps(num_bytes: int, seconds: float) -> float:
    """Compute throughput in MB/s (megabytes, base-10 definition: 1 MB = 10^6 bytes)."""
    if seconds <= 0:
        return 0.0
    return (num_bytes / seconds) / 1e6


def median_time(fn, runs: int) -> float:
    """Call `fn` `runs` times and return the median wall-clock time in seconds."""
    ts: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def benchmark(
    block_sizes: list[int],
    runs: int,
    cpu_workers: int,
    cpu_jit: bool,
    cpu_impl: str,
    gpu_variant: str,
    key_modes: list[str],
    mode: str,
) -> list[BenchRow]:
    """Run the AES-128 benchmark for all (block_size, key_mode) combinations.

    The CPU is warmed up and timed once per block size.  All GPU key modes
    are run against the same CPU measurement to avoid redundant CPU work.
    CPU/GPU ciphertext equality is verified at each point to detect bugs.

    Args:
        block_sizes: List of AES block counts to benchmark.
        runs:        Number of timed runs per point; median is reported.
        cpu_workers: Numba thread count for the parallel CPU path.
        cpu_jit:     Whether to enable Numba JIT for the CPU path.
        cpu_impl:    Label for the CPU implementation (always "software").
        gpu_variant: GPU S-box variant (always "shared" for AES).
        key_modes:   List of GPU key-storage modes to evaluate.
        mode:        Encryption mode: "ecb" or "ctr".

    Returns:
        List of BenchRow measurements, one per (block_size, key_mode).
    """
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    cpu = AesCpuOptimized(use_numba=cpu_jit)

    if not has_cuda_gpu():
        raise RuntimeError("No CUDA GPU detected")

    # Instantiate one GPU cipher per key mode, all sharing the same key.
    gpus = {}
    for km in key_modes:
        g = AesGpuOptimized(variant=gpu_variant, key_mode=km)
        g.set_key(key)
        gpus[km] = g

    total_steps = len(block_sizes)
    rows: list[BenchRow] = []
    km_label = "+".join(key_modes)

    for step_i, nblocks in enumerate(block_sizes, 1):
        _fmt_n = f"{nblocks:,}"
        print(f"\r  [{mode.upper()} {gpu_variant}/{km_label}] "
              f"Step {step_i}/{total_steps}: {_fmt_n} blocks ...", end="", flush=True)
        nbytes = nblocks * 16
        plaintext = np.random.randint(0, 256, nbytes, dtype=np.uint8).tobytes()
        ctr_nonce = os.urandom(8)

        # Warm up the CPU JIT cache, then measure median time.
        if mode == "ecb":
            _ = cpu.encrypt_ecb(plaintext, key, workers=cpu_workers)
            cpu_t = median_time(lambda: cpu.encrypt_ecb(plaintext, key, workers=cpu_workers), runs)
            cpu_ct = cpu.encrypt_ecb(plaintext, key, workers=cpu_workers)
        else:
            _ = cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce)
            cpu_t = median_time(lambda: cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce), runs)
            cpu_ct = cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce)

        # Benchmark each GPU key mode against the shared CPU baseline.
        for km in key_modes:
            gpu = gpus[km]

            # Warm up the GPU kernel (first launch incurs JIT compilation overhead).
            if mode == "ecb":
                _ = gpu.encrypt_ecb(plaintext)
                _ = gpu.encrypt_ecb(plaintext)
            else:
                _ = gpu.encrypt_ctr(plaintext, nonce=ctr_nonce)
                _ = gpu.encrypt_ctr(plaintext, nonce=ctr_nonce)

            gpu_total_ts: list[float] = []
            gpu_kernel_ts: list[float] = []
            gpu_transfer_ts: list[float] = []
            gpu_ct = None

            for _ in range(runs):
                if mode == "ecb":
                    gpu_ct, timing = gpu.encrypt_ecb(plaintext)
                else:
                    gpu_ct, timing = gpu.encrypt_ctr(plaintext, nonce=ctr_nonce)
                gpu_total_ts.append(timing.total_seconds)
                gpu_kernel_ts.append(timing.kernel_seconds)
                gpu_transfer_ts.append(timing.h2d_d2h_seconds)

            # Verify CPU and GPU outputs match (detects any correctness regression).
            if cpu_ct != gpu_ct:
                raise RuntimeError(f"CPU/GPU mismatch at {nblocks} blocks (mode={mode}, key_mode={km})")

            gpu_total = statistics.median(gpu_total_ts)
            gpu_kernel = statistics.median(gpu_kernel_ts)
            gpu_transfer = statistics.median(gpu_transfer_ts)

            rows.append(
                BenchRow(
                    mode=mode,
                    gpu_variant=gpu_variant,
                    key_mode=km,
                    cpu_workers=cpu_workers,
                    cpu_impl=cpu_impl,
                    blocks=nblocks,
                    bytes_total=nbytes,
                    cpu_seconds=cpu_t,
                    cpu_mbps=throughput_mbps(nbytes, cpu_t),
                    gpu_total_seconds=gpu_total,
                    gpu_kernel_seconds=gpu_kernel,
                    gpu_transfer_seconds=gpu_transfer,
                    gpu_total_mbps=throughput_mbps(nbytes, gpu_total),
                    gpu_kernel_mbps=throughput_mbps(nbytes, gpu_kernel),
                    speedup_total=(cpu_t / gpu_total) if gpu_total > 0 else 0.0,
                    speedup_kernel_only=(cpu_t / gpu_kernel) if gpu_kernel > 0 else 0.0,
                )
            )

    print()   # end the progress line with a newline
    return rows


def write_csv(rows: list[BenchRow], output_csv: Path) -> None:
    """Write all benchmark rows to a CSV file, creating parent directories if needed."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "gpu_variant", "key_mode", "cpu_workers", "cpu_impl",
            "blocks", "bytes_total", "cpu_seconds", "cpu_mbps",
            "gpu_total_seconds", "gpu_kernel_seconds", "gpu_transfer_seconds",
            "gpu_total_mbps", "gpu_kernel_mbps", "speedup_total", "speedup_kernel_only",
        ])
        for r in rows:
            writer.writerow([
                r.mode, r.gpu_variant, r.key_mode, r.cpu_workers, r.cpu_impl,
                r.blocks, r.bytes_total,
                f"{r.cpu_seconds:.9f}", f"{r.cpu_mbps:.6f}",
                f"{r.gpu_total_seconds:.9f}", f"{r.gpu_kernel_seconds:.9f}",
                f"{r.gpu_transfer_seconds:.9f}",
                f"{r.gpu_total_mbps:.6f}", f"{r.gpu_kernel_mbps:.6f}",
                f"{r.speedup_total:.6f}", f"{r.speedup_kernel_only:.6f}",
            ])


def print_summary(rows: list[BenchRow], cpu_workers: int, cpu_jit: bool, cpu_impl: str, gpu_variant: str, mode: str) -> None:
    """Print system environment info and a formatted throughput/speedup table."""
    print("\n=== Environment ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"CPU logical cores: {os.cpu_count() or 1}")
    print(f"CPU impl: {cpu_impl}, workers={cpu_workers}, jit={'on' if cpu_jit else 'off'}")
    print(f"GPU backend: Numba CUDA (JIT-compiled kernels)")

    print(f"\n=== AES CPU vs GPU Summary (Mode: {mode}, GPU Variant: {gpu_variant}) ===")

    for km in ("global", "constkeys"):
        km_rows = [r for r in rows if r.key_mode == km]
        if not km_rows:
            continue
        print(f"\nKey Mode: {km}")
        print(f"{'Blocks':>12} {'CPU MB/s':>12} {'GPU total MB/s':>16} {'GPU kernel MB/s':>17} {'Speedup(total)':>16} {'Speedup(kernel)':>17}")
        for r in km_rows:
            print(f"{r.blocks:12d} {r.cpu_mbps:12.3f} {r.gpu_total_mbps:16.3f} {r.gpu_kernel_mbps:17.3f} {r.speedup_total:16.3f} {r.speedup_kernel_only:17.3f}")


def main() -> None:
    """Parse CLI arguments and run the full AES-128 benchmark matrix."""
    parser = argparse.ArgumentParser(description="AES-128 CPU vs GPU benchmark (ECB/CTR)")
    parser.add_argument("--blocks", type=str, default="1024,16384,65536,262144,1048576,4194304,10485760,52428800,104857600")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parent.parent / "results" / "aes_benchmark.csv"))
    parser.add_argument("--cpu-workers", type=str, default="8", help="Comma-separated worker counts")
    parser.add_argument("--cpu-jit", dest="cpu_jit", action="store_true", help="Enable Numba JIT for CPU path (default: on)")
    parser.add_argument("--no-cpu-jit", dest="cpu_jit", action="store_false", help="Disable Numba JIT for CPU path")
    parser.set_defaults(cpu_jit=True)
    parser.add_argument("--cpu-impl", type=str, default="software",
                        help="Comma-separated CPU implementation labels (default: software)")
    parser.add_argument("--gpu-variant", type=str, choices=["shared"], default="shared")
    parser.add_argument("--key-mode", type=str, choices=["global", "constkeys", "both"], default="both")
    parser.add_argument("--mode", type=str, choices=["ecb", "ctr", "both"], default="both")
    args = parser.parse_args()

    block_sizes = parse_block_sizes(args.blocks)
    cpu_workers_list = [max(1, w) for w in parse_int_list(args.cpu_workers)]
    cpu_impl_list = parse_csv_list(args.cpu_impl)
    valid_cpu_impls = {"software"}
    for impl in cpu_impl_list:
        if impl not in valid_cpu_impls:
            raise ValueError(f"invalid cpu impl '{impl}', expected one of: {sorted(valid_cpu_impls)}")

    gpu_variants = ["shared"]
    key_modes = ["global", "constkeys"] if args.key_mode == "both" else [args.key_mode]
    modes = ["ecb", "ctr"] if args.mode == "both" else [args.mode]

    all_rows: list[BenchRow] = []
    for workers in cpu_workers_list:
        for impl in cpu_impl_list:
            for variant in gpu_variants:
                for mode in modes:
                    rows = benchmark(
                        block_sizes=block_sizes,
                        runs=args.runs,
                        cpu_workers=workers,
                        cpu_jit=bool(args.cpu_jit),
                        cpu_impl=impl,
                        gpu_variant=variant,
                        key_modes=key_modes,
                        mode=mode,
                    )
                    all_rows.extend(rows)

    write_csv(all_rows, Path(args.csv))

    for workers in cpu_workers_list:
        for impl in cpu_impl_list:
            for variant in gpu_variants:
                for mode in modes:
                    subset = [
                        r for r in all_rows
                        if r.cpu_workers == workers
                        and r.cpu_impl == impl
                        and r.gpu_variant == variant
                        and r.mode == mode
                    ]
                    print_summary(subset, workers, bool(args.cpu_jit), impl, variant, mode)


if __name__ == "__main__":
    main()
