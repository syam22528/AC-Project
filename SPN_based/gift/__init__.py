"""GIFT-64-128 cipher implementations: CPU and GPU backends.

Exports optimized CPU/GPU classes and shared key schedule helpers.
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
