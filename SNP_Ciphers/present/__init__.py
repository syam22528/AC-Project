"""PRESENT-128 lightweight block cipher implementations: CPU and GPU.

Supports ECB and CTR modes for both single-thread and parallel execution.
Optimized for benchmarking against other SPN ciphers.
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
