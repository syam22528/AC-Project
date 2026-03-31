from __future__ import annotations

"""AES-128 correctness verification for CPU and GPU implementations.

Verifies:
1. NIST test vector (standard reference)
2. CPU/GPU consistency on random data for all GPU variant combinations
"""

import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from aes.cpu import AesCpuOptimized
    from aes.gpu import AesGpuOptimized, has_cuda_gpu
else:
    from .cpu import AesCpuOptimized
    from .gpu import AesGpuOptimized, has_cuda_gpu


def main() -> None:
    """Verify AES correctness: NIST vector + random CPU/GPU consistency."""
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    expected = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

    cpu = AesCpuOptimized()
    cpu_ct = cpu.encrypt_ecb(pt, key, workers=1)
    if cpu_ct != expected:
        raise RuntimeError("CPU AES-128 failed NIST vector")

    if not has_cuda_gpu():
        print("CUDA GPU not detected; CPU vector check passed")
        return

    # Test random data with all GPU variant/key-mode combinations
    random_data = os.urandom(16 * 8192)
    key2 = os.urandom(16)

    for variant in ("shared", "bitsliced"):
        for key_mode in ("global", "constkeys"):
            gpu = AesGpuOptimized(block_size=256, variant=variant, key_mode=key_mode)
            gpu.set_key(key)

            # Verify NIST vector with this GPU variant
            gpu_ct, _ = gpu.encrypt_ecb(pt)
            if gpu_ct != expected:
                raise RuntimeError(f"GPU optimized ({variant}, key_mode={key_mode}) kernel failed NIST vector")

            # Verify CPU/GPU consistency on random data
            cpu_ref = cpu.encrypt_ecb(random_data, key2, workers=1)
            gpu.set_key(key2)
            gpu_ct, _ = gpu.encrypt_ecb(random_data)
            if gpu_ct != cpu_ref:
                raise RuntimeError(f"CPU/GPU mismatch for optimized ({variant}, key_mode={key_mode}) kernel")

    print("All correctness checks passed (AES CPU and GPU)")


if __name__ == "__main__":
    main()
