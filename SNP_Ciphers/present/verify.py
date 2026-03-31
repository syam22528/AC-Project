from __future__ import annotations

"""PRESENT-128 correctness verification for CPU and GPU implementations.

Checks:
1. NIST-style known-answer test for PRESENT-128 (all-zero plaintext and key).
2. CPU/GPU consistency on random data for both GPU S-box variants
   (table and bitsliced), confirming both kernels produce identical output
   to the CPU reference implementation.
"""

import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from cpu import PresentCpuOptimized
    from gpu import PresentGpuOptimized, has_cuda_gpu
else:
    from .cpu import PresentCpuOptimized
    from .gpu import PresentGpuOptimized, has_cuda_gpu


def main() -> None:
    """Run the PRESENT-128 known-answer test then CPU/GPU consistency checks."""
    cpu = PresentCpuOptimized(use_numba=True)

    # Published PRESENT-128 test vector: all-zero plaintext and key.
    pt = bytes.fromhex("0000000000000000")
    key128 = bytes.fromhex("00000000000000000000000000000000")
    expected128 = bytes.fromhex("96db702a2e6900af")
    ct128 = cpu.encrypt_ecb(pt, key128, workers=1)
    if ct128 != expected128:
        raise RuntimeError(f"PRESENT-128 vector failed: got={ct128.hex()} expected={expected128.hex()}")

    if not has_cuda_gpu():
        print("PRESENT CPU correctness checks passed (GPU not detected)")
        return

    # Cross-check CPU and GPU outputs on a large random payload.
    random_data = os.urandom(8 * 32768)

    for variant in ("table", "bitsliced"):
        gpu = PresentGpuOptimized(block_size=256, variant=variant)

        # Verify both GPU variants produce the same output as the CPU for the known key.
        gpu.set_key(key128)
        gpu_ct128, _ = gpu.encrypt_ecb(random_data)
        cpu_ct128 = cpu.encrypt_ecb(random_data, key128, workers=1)
        if gpu_ct128 != cpu_ct128:
            raise RuntimeError(f"CPU/GPU mismatch for PRESENT-128 (variant={variant})")

    print("All PRESENT correctness checks passed (CPU and GPU: table + bitsliced)")


if __name__ == "__main__":
    main()
