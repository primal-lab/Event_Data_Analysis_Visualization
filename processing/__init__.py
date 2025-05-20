"""
Event processing module.
Contains functions for event tensor generation and diffusion processing.
"""

from .event_tensor import build_event_tensor, process_window
from .diffusion import heat_kernel, generate_heat_kernel_3d_np

__all__ = [
    'build_event_tensor',
    'process_window',
    'heat_kernel',
    'generate_heat_kernel_3d_np',
] 