"""PRESENT-128 package: CPU and GPU cipher implementations.

Exports the CPU class (Numba JIT, delta-swap pLayer), the GPU class
(CUDA kernels for bitsliced and table S-box variants), and the key
schedule function used by both backends.
"""

from .cpu import PresentCpuOptimized
from .gpu import PresentGpuOptimized, has_cuda_gpu
from .common import generate_round_keys

__all__ = [
    "PresentCpuOptimized",
    "PresentGpuOptimized",
    "generate_round_keys",
    "has_cuda_gpu",
]
