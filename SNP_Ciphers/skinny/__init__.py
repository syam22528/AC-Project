"""SKINNY-64-128 package: CPU and GPU cipher implementations.

Exports the CPU class (Numba JIT, table S-box), the GPU class
(CUDA kernels for table and bitsliced S-box variants), and the key
schedule helper shared by both backends.
"""

from .cpu import SkinnyCpuOptimized
from .gpu import SkinnyGpuOptimized, has_cuda_gpu
from .common import SKINNY_ROUNDS, generate_round_subkeys

__all__ = [
    "SkinnyCpuOptimized",
    "SkinnyGpuOptimized",
    "SKINNY_ROUNDS",
    "generate_round_subkeys",
    "has_cuda_gpu",
]
