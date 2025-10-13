"""Helpers for the ``daniele_experiment`` package."""

import torch


def get_device(device: str = "auto") -> str:
    """Auto-detect the best available PyTorch device with fallback logic.
    
    Args:
        device: Device preference. If "auto", will detect best available device.
                Otherwise returns the specified device.
    
    Returns:
        Device string: "mps", "cuda", or "cpu"
        
    Device priority:
        1. MPS (Metal Performance Shaders) - for Apple Silicon Macs
        2. CUDA - for NVIDIA GPUs
        3. CPU - fallback for all other cases
    """
    if device != "auto":
        return device
    
    # Check for MPS (Apple Silicon Macs)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    
    # Fallback to CPU
    return "cpu"


__all__ = [
    "get_device",
]
