from __future__ import annotations

"""GIFT-64-128 CPU vs GPU benchmark in ECB and CTR modes.

Measures throughput (MB/s) and GPU speedup for the table and bitsliced
S-box GPU variants.  The CPU is timed once per block size; all GPU variants
are compared against that same baseline to avoid redundant CPU work.

Results are written to CSV and printed as a formatted table.
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

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from cpu import GiftCpuOptimized
    from gpu import GiftGpuOptimized, has_cuda_gpu
else:
    from .cpu import GiftCpuOptimized
    from .gpu import GiftGpuOptimized, has_cuda_gpu


@dataclass
class BenchRow:
    """One row of benchmark results for a single (block_count, mode, variant) point."""
    mode: str
    cpu_workers: int
    variant: str
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


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated string of integers into a list of ints."""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def throughput_mbps(num_bytes: int, seconds: float) -> float:
    """Compute throughput in MB/s (base-10: 1 MB = 10^6 bytes)."""
    if seconds <= 0:
        return 0.0
    return (num_bytes / seconds) / 1e6


def median_time(fn, runs: int) -> float:
    """Call `fn` `runs` times and return the median wall-clock time in seconds."""
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def benchmark(
    block_sizes: list[int],
    runs: int,
    cpu_workers: int,
    gpu_variants: list[str],
    mode: str,
) -> list[BenchRow]:
    """Run the GIFT-64-128 benchmark for all (block_size, variant) combinations.

    The CPU is warmed up and timed once per block size.  All GPU variants
    share the same CPU measurement.  CPU/GPU ciphertext equality is verified
    at every point.

    Args:
        block_sizes:  List of GIFT-64 block counts to benchmark.
        runs:         Number of timed runs per point; median is reported.
        cpu_workers:  Numba thread count for the parallel CPU path.
        gpu_variants: GPU S-box variants to evaluate ("table", "bitsliced").
        mode:         Encryption mode: "ecb" or "ctr".

    Returns:
        List of BenchRow measurements.
    """
    if not has_cuda_gpu():
        raise RuntimeError("No CUDA GPU detected")

    key = os.urandom(16)
    cpu = GiftCpuOptimized(use_numba=True)

    # Instantiate one GPU cipher per variant, all using the same key.
    gpus = {}
    for v in gpu_variants:
        g = GiftGpuOptimized(block_size=256, variant=v)
        g.set_key(key)
        gpus[v] = g

    rows: list[BenchRow] = []
    total_steps = len(block_sizes)
    variant_label = "+".join(gpu_variants)

    for step_i, nblocks in enumerate(block_sizes, 1):
        _fmt_n = f"{nblocks:,}"
        print(f"\r  [{mode.upper()} {variant_label}] "
              f"Step {step_i}/{total_steps}: {_fmt_n} blocks ...", end="", flush=True)
        nbytes = nblocks * 8
        plaintext = os.urandom(nbytes)
        ctr_nonce = os.urandom(4)

        # Warm up the CPU JIT cache, then measure median throughput.
        if mode == "ecb":
            _ = cpu.encrypt_ecb(plaintext, key, workers=cpu_workers)
            cpu_t = median_time(lambda: cpu.encrypt_ecb(plaintext, key, workers=cpu_workers), runs)
            cpu_ct = cpu.encrypt_ecb(plaintext, key, workers=cpu_workers)
        else:
            _ = cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce)
            cpu_t = median_time(lambda: cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce), runs)
            cpu_ct = cpu.encrypt_ctr(plaintext, key, workers=cpu_workers, nonce=ctr_nonce)

        # Benchmark each GPU variant against the shared CPU baseline.
        for variant in gpu_variants:
            gpu = gpus[variant]

            # Warm up the GPU kernel before timed runs.
            if mode == "ecb":
                _ = gpu.encrypt_ecb(plaintext)
            else:
                _ = gpu.encrypt_ctr(plaintext, nonce=ctr_nonce)

            gpu_total_ts = []
            gpu_kernel_ts = []
            gpu_transfer_ts = []
            gpu_ct = None

            for _ in range(runs):
                if mode == "ecb":
                    gpu_ct, timing = gpu.encrypt_ecb(plaintext)
                else:
                    gpu_ct, timing = gpu.encrypt_ctr(plaintext, nonce=ctr_nonce)
                gpu_total_ts.append(timing.total_seconds)
                gpu_kernel_ts.append(timing.kernel_seconds)
                gpu_transfer_ts.append(timing.h2d_d2h_seconds)

            # Correctness gate: GPU output must match CPU output.
            if cpu_ct != gpu_ct:
                raise RuntimeError(f"CPU/GPU mismatch at {nblocks} blocks (mode={mode}, variant={variant})")

            gpu_total = statistics.median(gpu_total_ts)
            gpu_kernel = statistics.median(gpu_kernel_ts)
            gpu_transfer = statistics.median(gpu_transfer_ts)

            rows.append(
                BenchRow(
                    mode=mode,
                    cpu_workers=cpu_workers,
                    variant=variant,
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

    print()   # end progress line
    return rows


def write_csv(rows: list[BenchRow], output_csv: Path) -> None:
    """Write all benchmark rows to a CSV file, creating parent directories if needed."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "cpu_workers", "variant", "blocks", "bytes_total",
            "cpu_seconds", "cpu_mbps", "gpu_total_seconds", "gpu_kernel_seconds",
            "gpu_transfer_seconds", "gpu_total_mbps", "gpu_kernel_mbps",
            "speedup_total", "speedup_kernel_only",
        ])
        for r in rows:
            w.writerow([
                r.mode, r.cpu_workers, r.variant, r.blocks, r.bytes_total,
                f"{r.cpu_seconds:.9f}", f"{r.cpu_mbps:.6f}",
                f"{r.gpu_total_seconds:.9f}", f"{r.gpu_kernel_seconds:.9f}",
                f"{r.gpu_transfer_seconds:.9f}",
                f"{r.gpu_total_mbps:.6f}", f"{r.gpu_kernel_mbps:.6f}",
                f"{r.speedup_total:.6f}", f"{r.speedup_kernel_only:.6f}",
            ])


def print_summary(rows: list[BenchRow], cpu_workers: int, mode: str) -> None:
    """Print system environment info and a formatted throughput/speedup table."""
    print("\n=== Environment ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"CPU logical cores: {os.cpu_count() or 1}")

    print("\n=== GIFT-64-128 CPU vs GPU Summary ===")
    print(f"Config: mode={mode}, CPU workers={cpu_workers}")

    variant_desc = {
        "table":     "table-lookup S-box (constant memory)",
        "bitsliced": "bitsliced tableless S-box (boolean logic)",
    }

    for variant in ("table", "bitsliced"):
        vrows = [r for r in rows if r.variant == variant]
        if not vrows:
            continue
        print(f"\nGPU S-box variant: {variant_desc.get(variant, variant)}")
        print(f"{'Blocks':>12} {'CPU MB/s':>12} {'GPU total MB/s':>16} {'GPU kernel MB/s':>17} {'Speedup(total)':>16} {'Speedup(kernel)':>17}")
        for r in vrows:
            print(f"{r.blocks:12d} {r.cpu_mbps:12.3f} {r.gpu_total_mbps:16.3f} {r.gpu_kernel_mbps:17.3f} {r.speedup_total:16.3f} {r.speedup_kernel_only:17.3f}")


def main() -> None:
    """Parse CLI arguments and run the full GIFT-64-128 benchmark matrix."""
    ap = argparse.ArgumentParser(
        description="GIFT-64-128 CPU vs GPU benchmark with configurable S-box strategy (ECB/CTR)"
    )
    ap.add_argument("--cpu-workers", type=str, default="8", help="Comma-separated worker counts")
    ap.add_argument("--gpu-variant", type=str, default="both", choices=["table", "bitsliced", "both"],
                    help="S-box implementation: table, bitsliced, or both")
    ap.add_argument("--blocks", type=str, default="1024,16384,65536,262144,1048576,4194304,10485760,52428800,104857600")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--mode", type=str, default="both", choices=["ecb", "ctr", "both"])
    ap.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parent.parent / "results" / "gift_benchmark.csv"))
    args = ap.parse_args()

    workers = [max(1, w) for w in parse_int_list(args.cpu_workers)]
    variants = ["table", "bitsliced"] if args.gpu_variant == "both" else [args.gpu_variant]
    modes = ["ecb", "ctr"] if args.mode == "both" else [args.mode]

    rows: list[BenchRow] = []
    for cpu_workers in workers:
        for mode in modes:
            rows.extend(
                benchmark(
                    block_sizes=parse_block_sizes(args.blocks),
                    runs=args.runs,
                    cpu_workers=cpu_workers,
                    gpu_variants=variants,
                    mode=mode,
                )
            )

    write_csv(rows, Path(args.csv))

    for cpu_workers in workers:
        for mode in modes:
            subset = [r for r in rows if r.cpu_workers == cpu_workers and r.mode == mode]
            print_summary(subset, cpu_workers, mode)


if __name__ == "__main__":
    main()
