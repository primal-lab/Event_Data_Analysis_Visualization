"""
Event processing module.
Contains functions for event tensor generation and diffusion processing.
"""

from .diffusion import heat_kernel, generate_heat_kernel_3d_np

__all__ = [
    'heat_kernel',
    'generate_heat_kernel_3d_np',
] 