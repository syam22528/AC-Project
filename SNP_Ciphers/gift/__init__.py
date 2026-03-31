"""GIFT-64-128 package: CPU and GPU cipher implementations.

Exports the CPU class (Numba JIT, table S-box), the GPU class
(CUDA kernels for table and bitsliced S-box variants), and the key
schedule helper used by both backends.
"""

from .cpu import GiftCpuOptimized
from .gpu import GiftGpuOptimized, has_cuda_gpu
from .common import GIFT_ROUNDS, generate_round_masks

__all__ = [
    "GiftCpuOptimized",
    "GiftGpuOptimized",
    "GIFT_ROUNDS",
    "generate_round_masks",
    "has_cuda_gpu",
]
