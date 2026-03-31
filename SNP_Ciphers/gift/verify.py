from __future__ import annotations

"""Correctness checks for GIFT CPU and GPU implementations.

Validates known-answer vectors and cross-checks GPU output against CPU.
"""

import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from cpu import GiftCpuOptimized
    from gpu import GiftGpuOptimized, has_cuda_gpu
else:
    from .cpu import GiftCpuOptimized
    from .gpu import GiftGpuOptimized, has_cuda_gpu


def main() -> None:
    """Run known-vector checks and CPU/GPU consistency checks."""
    vectors = [
        (
            bytes.fromhex("0000000000000000"),
            bytes.fromhex("00000000000000000000000000000000"),
            bytes.fromhex("f62bc3ef34f775ac"),
        ),
        (
            bytes.fromhex("fedcba9876543210"),
            bytes.fromhex("fedcba9876543210fedcba9876543210"),
            bytes.fromhex("c1b71f66160ff587"),
        ),
        (
            bytes.fromhex("c450c7727a9b8a7d"),
            bytes.fromhex("bd91731eb6bc2713a1f9f6ffc75044e7"),
            bytes.fromhex("e3272885fa94ba8b"),
        ),
    ]

    cpu = GiftCpuOptimized(use_numba=True)
    for pt, key, exp in vectors:
        ct = cpu.encrypt_ecb(pt, key, workers=1)
        if ct != exp:
            raise RuntimeError(
                f"GIFT known vector failed: got={ct.hex()} expected={exp.hex()}"
            )

    if not has_cuda_gpu():
        print("GIFT CPU correctness checks passed (GPU not detected)")
        return

    random_data = os.urandom(8 * 32768)
    key = os.urandom(16)

    cpu_ref = GiftCpuOptimized(use_numba=True)
    ref_ct = cpu_ref.encrypt_ecb(random_data, key, workers=1)

    for variant in ("table", "bitsliced"):
        gpu = GiftGpuOptimized(block_size=256, variant=variant)
        gpu.set_key(key)
        gpu_ct, _ = gpu.encrypt_ecb(random_data)
        if gpu_ct != ref_ct:
            raise RuntimeError(f"CPU/GPU mismatch for GIFT (variant={variant})")

    print("All GIFT correctness checks passed (CPU and GPU: table + bitsliced)")


if __name__ == "__main__":
    main()
