"""SKINNY-64-128 cipher implementations: CPU and GPU backends.

Exports optimized CPU/GPU classes plus shared key schedule helpers.
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
