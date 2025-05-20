"""
Heatmap generation utilities.
Contains functions for creating heatmaps from event data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import torch

def generate_heatmap(
    event_data: Union[np.ndarray, torch.Tensor],
    height: int,
    width: int,
    sigma: float = 1.0,
    normalize: bool = True,
    cmap: str = 'inferno'
) -> np.ndarray:
    """
    Generate a heatmap from event data.
    
    Args:
        event_data (Union[np.ndarray, torch.Tensor]): Event data
        height (int): Height of the output heatmap
        width (int): Width of the output heatmap
        sigma (float): Standard deviation for Gaussian blur
        normalize (bool): Whether to normalize the heatmap
        cmap (str): Colormap to use
        
    Returns:
        np.ndarray: Generated heatmap
    """
    # Convert to numpy if needed
    if torch.is_tensor(event_data):
        event_data = event_data.cpu().numpy()
    
    # Create empty heatmap
    heatmap = np.zeros((height, width))
    
    # Add events to heatmap
    if len(event_data) > 0:
        x = event_data[:, 0].astype(int)
        y = event_data[:, 1].astype(int)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[valid], y[valid]
        
        # Add events
        heatmap[y, x] += 1
    
    # Apply Gaussian blur
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    
    # Normalize
    if normalize and np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

def heat_kernel(X, Y, t, k):
    """Generate a 2D heat kernel."""
    return np.exp(-(X**2 + Y**2) / (4 * k * t)) / (4 * np.pi * k * t)

def generate_heat_kernel_3d_np(T, H, W, k=0.05):
    """Generate a 3D heat kernel for temporal-spatial diffusion."""
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel(X, Y, t/T, k)
        kernel[t - 1] /= kernel[t - 1].sum()
    return kernel

def precompute_heat_kernel_and_derivatives(size=31, alpha=1.0, t=1.0):
    """
    Precompute heat kernel and its spatial derivatives.
    
    Args:
        size (int): Size of the kernel
        alpha (float): Diffusion coefficient
        t (float): Time parameter
        
    Returns:
        tuple: (dH_dx, dH_dy) - Spatial derivatives of the heat kernel
    """
    r = size // 2
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, y, indexing='xy')

    denom = 4 * np.pi * alpha * t
    H = (1.0 / denom) * np.exp(-(X**2 + Y**2) / (4 * alpha * t))

    dH_dx = H * (-X / (2 * alpha * t))
    dH_dy = H * (-Y / (2 * alpha * t))
    return dH_dx, dH_dy

def apply_kernel_gradients(frame, dH_dx, dH_dy):
    """
    Apply heat kernel gradients to frame.
    
    Args:
        frame (np.ndarray): Input frame
        dH_dx (np.ndarray): x-derivative of heat kernel
        dH_dy (np.ndarray): y-derivative of heat kernel
        
    Returns:
        tuple: (grad_x, grad_y) - Gradient maps
    """
    grad_x = convolve2d(frame, dH_dx, mode='same', boundary='symm')
    grad_y = convolve2d(frame, dH_dy, mode='same', boundary='symm')
    return grad_x, grad_y 