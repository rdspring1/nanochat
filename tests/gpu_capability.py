"""
GPU capability detection for hardware-specific features.

This module centralizes GPU compute capability detection and feature support
checks for different GPU architectures (Blackwell, Hopper, etc.).
"""

import torch


def get_gpu_capability():
    """
    Get GPU compute capability.

    Returns:
        tuple[int, int] | None: (major, minor) compute capability, or None if CUDA not available

    Examples:
        - Blackwell B200: (10, 0) = sm100
        - Hopper H100: (9, 0) = sm90
        - Ada L40S: (8, 9) = sm89
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def is_blackwell_gpu():
    """
    Check if current GPU is Blackwell architecture (B200, B200 Ultra).

    Returns:
        bool: True if Blackwell GPU (sm100, compute capability 10.0), False otherwise
    """
    capability = get_gpu_capability()
    if capability is None:
        return False
    major, _ = capability
    return major == 10  # Blackwell is sm100
