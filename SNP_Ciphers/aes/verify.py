from __future__ import annotations

"""AES-128 correctness verification for CPU and GPU implementations.

Runs two checks:
1. NIST FIPS 197 known-answer test vector to confirm correct AES-128 output.
2. CPU/GPU consistency on random plaintext for all GPU key-mode combinations
   (global and constkeys), to ensure both backends produce identical ciphertext.
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
    """Run NIST vector check then CPU/GPU consistency checks for all key modes."""
    # NIST FIPS 197 Appendix B test vector for AES-128.
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

    # Verify GPU output against the CPU reference on a larger random payload.
    random_data = os.urandom(16 * 8192)
    key2 = os.urandom(16)

    for key_mode in ("global", "constkeys"):
        gpu = AesGpuOptimized(block_size=256, variant="shared", key_mode=key_mode)
        gpu.set_key(key)

        # Confirm the NIST vector is correct for this key mode.
        gpu_ct, _ = gpu.encrypt_ecb(pt)
        if gpu_ct != expected:
            raise RuntimeError(f"GPU (key_mode={key_mode}) failed NIST vector")

        # Cross-check CPU and GPU outputs on random data with a fresh random key.
        cpu_ref = cpu.encrypt_ecb(random_data, key2, workers=1)
        gpu.set_key(key2)
        gpu_ct, _ = gpu.encrypt_ecb(random_data)
        if gpu_ct != cpu_ref:
            raise RuntimeError(f"CPU/GPU mismatch for AES (key_mode={key_mode})")

    print("All correctness checks passed (AES CPU and GPU)")


if __name__ == "__main__":
    main()
