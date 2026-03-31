from __future__ import annotations

"""SKINNY-64-128 correctness verification for CPU and GPU implementations.

Runs two checks:
1. A known-answer test against a published SKINNY-64-128 test vector
   (from the skinny-c reference test suite) to confirm correct CPU output.
2. CPU/GPU consistency: encrypts a large random payload on the CPU and
   compares the result against both GPU S-box variants (table and bitsliced).
"""

import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from cpu import SkinnyCpuOptimized
    from gpu import SkinnyGpuOptimized, has_cuda_gpu
else:
    from .cpu import SkinnyCpuOptimized
    from .gpu import SkinnyGpuOptimized, has_cuda_gpu


def main() -> None:
    """Run the SKINNY-64-128 known-vector check then CPU/GPU consistency checks."""
    # Published SKINNY-64-128 test vector from the skinny-c reference implementation.
    vectors = [
        (
            bytes.fromhex("cf16cfe8fd0f98aa"),   # plaintext
            bytes.fromhex("9eb93640d088da6376a39d1c8bea71e1"),   # 128-bit tweakey
            bytes.fromhex("6ceda1f43de92b9e"),   # expected ciphertext
        ),
    ]

    cpu = SkinnyCpuOptimized(use_numba=True)
    for pt, key, exp in vectors:
        ct = cpu.encrypt_ecb(pt, key, workers=1)
        if ct != exp:
            raise RuntimeError(
                f"SKINNY vector failed: got={ct.hex()} expected={exp.hex()}"
            )

    if not has_cuda_gpu():
        print("SKINNY CPU correctness checks passed (GPU not detected)")
        return

    # Cross-check CPU and GPU outputs on a large random payload.
    random_data = os.urandom(8 * 32768)
    key = os.urandom(16)

    cpu_ref = SkinnyCpuOptimized(use_numba=True)
    ref_ct = cpu_ref.encrypt_ecb(random_data, key, workers=1)

    for variant in ("table", "bitsliced"):
        gpu = SkinnyGpuOptimized(block_size=256, variant=variant)
        gpu.set_key(key)
        gpu_ct, _ = gpu.encrypt_ecb(random_data)
        if gpu_ct != ref_ct:
            raise RuntimeError(f"CPU/GPU mismatch for SKINNY (variant={variant})")

    print("All SKINNY correctness checks passed (CPU and GPU: table + bitsliced)")


if __name__ == "__main__":
    main()
