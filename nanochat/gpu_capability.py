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


def is_hopper_gpu():
    """
    Check if current GPU is Hopper architecture (H100, H200).

    Returns:
        bool: True if Hopper GPU (sm90, compute capability 9.0), False otherwise
    """
    capability = get_gpu_capability()
    if capability is None:
        return False
    major, _ = capability
    return major == 9  # Hopper is sm90


def supports_mxfp8():
    """
    Check if MXFP8 (Microscaling FP8) training is supported.

    MXFP8 requires Blackwell B200+ GPU and CUDA 12.8+.

    Returns:
        bool: True if MXFP8 is supported on current hardware
    """
    return is_blackwell_gpu()


def supports_nvfp4():
    """
    Check if NVFP4 (4-bit floating point) training is supported.

    NVFP4 requires Blackwell B200+ GPU and CUDA 12.8+.
    Note: NVFP4 is experimental in torchao.

    Returns:
        bool: True if NVFP4 is supported on current hardware
    """
    return is_blackwell_gpu()


def get_gpu_name():
    """
    Get human-readable GPU name for logging.

    Returns:
        str: GPU name (e.g., "NVIDIA H100 80GB HBM3") or "No CUDA GPU" if unavailable
    """
    if not torch.cuda.is_available():
        return "No CUDA GPU"
    return torch.cuda.get_device_name(0)


def get_architecture_name():
    """
    Get GPU architecture name based on compute capability.

    Returns:
        str: Architecture name (e.g., "Blackwell", "Hopper", "Ada", "Unknown")
    """
    capability = get_gpu_capability()
    if capability is None:
        return "Unknown"

    major, minor = capability

    # Map compute capabilities to architecture names
    if major == 10:
        return "Blackwell"  # sm100
    elif major == 9:
        return "Hopper"  # sm90
    elif major == 8 and minor == 9:
        return "Ada"  # sm89
    elif major == 8 and minor == 6:
        return "Ampere"  # sm86 (A100)
    elif major == 8 and minor == 0:
        return "Ampere"  # sm80 (A100)
    elif major == 7 and minor == 5:
        return "Turing"  # sm75
    elif major == 7 and minor == 0:
        return "Volta"  # sm70
    else:
        return f"Unknown (sm{major}{minor})"
