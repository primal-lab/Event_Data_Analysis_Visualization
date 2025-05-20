"""
Diffusion processing utilities.
Contains functions for applying diffusion and generating diffusion kernels.
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # Use non-interactive backend
import imageio.v3 as iio


def heat_kernel(x, y, t, k):
    denom = 4 * np.pi * k * (t + 1e-8)
    exponent = -(x**2 + y**2) / (4 * k * (t + 1e-8))
    return (1 / denom) * np.exp(exponent)

def generate_heat_kernel_3d_np(T, H, W, k=0.05):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel(X, Y, t/T, k) # This is equivalent to dividing 
        # the alpha value by T which seems to provide better visual results
        kernel[t - 1] /= kernel[t - 1].sum()
    return kernel

def apply_diffusion(state, kernel):
    """Apply diffusion using convolution."""
    from scipy.signal import convolve2d
    # Ensure kernel is 2D
    if len(kernel.shape) > 2:
        kernel = kernel[0]  # Take first slice if 3D
    # Normalize kernel
    kernel = kernel / kernel.sum()
    # Apply convolution
    return convolve2d(state, kernel, mode='same', boundary='wrap')

if __name__ == "__main__":
    # Parameters for visualization
    T = 200  # Number of time steps
    H, W = 7, 7  # Grid size
    k = 0.2  # Increased diffusion coefficient for faster spreading
    
    # Generate initial hot point
    initial_state = np.zeros((H, W), dtype=np.float32)
    center_y, center_x = H // 2, W // 2
    initial_state[center_y, center_x] = 1.0
    
    # Generate diffusion kernel
    kernel = generate_heat_kernel_3d_np(T, H, W, k=k)
    
    # Create frames for visualization
    frames = []
    current_state = initial_state.copy()
    
    # Create figure with custom style
    plt.style.use('dark_background')
    
    for t in tqdm(range(T)):
        # Apply diffusion using convolution
        current_state = apply_diffusion(current_state, kernel[t])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create heatmap
        im = ax.imshow(current_state, 
                      cmap='inferno',
                      interpolation='nearest',
                      vmin=0,
                      vmax=0.3)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add title with time step
        ax.set_title(f'Heat Diffusion at t = {t}', color='white', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])  # Remove alpha channel
        
        plt.close(fig)
    
    # Save as GIF
    os.makedirs("Plots", exist_ok=True)
    gif_path = "Plots/heat_diffusion_visualization.gif"
    iio.imwrite(gif_path, frames, duration=100)  # 20 FPS
    
    print(f"✅ GIF saved to {gif_path}")
    print(f"Visualization shows heat diffusion from a single hot point")
    print(f"Grid size: {H}x{W}, Time steps: {T}, Diffusion coefficient: {k}")