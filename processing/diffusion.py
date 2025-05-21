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

def heat_kernel_gradient(x, y, t, k):
    denom = 4 * np.pi * k * (t + 1e-8)
    exponent = -(x**2 + y**2) / (4 * k * (t + 1e-8))
    return (1 / denom) * np.exp(exponent) * (-1* x / (2 * k * (t + 1e-8)))

def heat_kernel_gradient_components(x, y, t, k):
    """Compute the x and y components of the heat kernel gradient."""
    denom = 4 * np.pi * k * (t + 1e-8)
    exponent = -(x**2 + y**2) / (4 * k * (t + 1e-8))
    common_factor = (1 / denom) * np.exp(exponent)
    dh_x = common_factor * (-1 * x / (2 * k * (t + 1e-8)))
    dh_y = common_factor * (-1 * y / (2 * k * (t + 1e-8)))
    return dh_x, dh_y

def compute_gradient_magnitude_and_angle(x, y, t, k):
    """Compute the magnitude and angle of the heat kernel gradient."""
    dh_x, dh_y = heat_kernel_gradient_components(x, y, t, k)
    magnitude = np.sqrt(dh_x**2 + dh_y**2)
    # print(magnitude)
    # Normalize magnitude to be between 0 and 1
    magnitude = magnitude / (np.min(magnitude) + 1e-8)
    angle = np.arctan2(dh_y, dh_x)
    return np.clip(magnitude, 0, 1), angle

def plot_gradient_quiver(x, y, t, k, ax=None):
    """Create a quiver plot of the heat kernel gradient."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    magnitude, angle = compute_gradient_magnitude_and_angle(x, y, t, k)
    print(magnitude)
    print(angle)
    dh_x = magnitude * np.cos(angle)
    dh_y = magnitude * np.sin(angle)
    ax.quiver(x, y, dh_x, dh_y, scale=1)  # Adjusted scale for better visibility
    ax.set_title(f'Heat Kernel Gradient Quiver Plot at t = {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

def generate_heat_kernel_3d_np(T, H, W, k=0.05):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel(X, Y, t/T, k)
        # Add small epsilon to prevent division by zero
        kernel_sum = kernel[t - 1].sum()
        if kernel_sum > 0:
            kernel[t - 1] /= kernel_sum
        else:
            kernel[t - 1] = np.zeros_like(kernel[t - 1])
            kernel[t - 1, cy, cx] = 1.0  # Set center point to 1 if all zeros
    return kernel

def generate_heat_kernel_gradient_3d_np(T, H, W, k=0.05):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel_gradient(X, Y, t/T, k)
        # Add small epsilon to prevent division by zero
        kernel_sum = kernel[t - 1].sum()
        if kernel_sum > 0:
            kernel[t - 1] /= kernel_sum
        else:
            kernel[t - 1] = np.zeros_like(kernel[t - 1])
            kernel[t - 1, cy, cx] = 1.0  # Set center point to 1 if all zeros
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
    kernel_gradient = generate_heat_kernel_gradient_3d_np(T, H, W, k=k)
    
    # Create frames for visualization
    frames = []
    frames_gradient = []
    current_state = initial_state.copy()
    current_state_gradient = initial_state.copy()
    
    # Create figure with custom style
    plt.style.use('dark_background')
    
    for t in tqdm(range(T)):
        # Apply diffusion using convolution
        current_state = apply_diffusion(current_state, kernel[t])
        current_state_gradient = apply_diffusion(current_state_gradient, kernel_gradient[t])
        
        # --- Regular diffusion visualization ---
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(current_state, 
                      cmap='inferno',
                      interpolation='nearest',
                      vmin=0,
                      vmax=0.3)
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        ax.set_title(f'Heat Diffusion at t = {t}', color='white', pad=20)
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3])  # Remove alpha channel
        plt.close(fig)

        # --- Gradient diffusion visualization ---
        fig_g, ax_g = plt.subplots(figsize=(8, 8))
        # Define X and Y for the quiver plot
        y = np.arange(H) - center_y
        x = np.arange(W) - center_x
        Y, X = np.meshgrid(y, x, indexing='ij')
        plot_gradient_quiver(X, Y, t/T, k, ax=ax_g)
        plt.tight_layout()
        fig_g.canvas.draw()
        frame_g = np.frombuffer(fig_g.canvas.buffer_rgba(), dtype=np.uint8)
        frame_g = frame_g.reshape(fig_g.canvas.get_width_height()[::-1] + (4,))
        frames_gradient.append(frame_g[:, :, :3])  # Remove alpha channel
        plt.close(fig_g)
    
    # Save as GIF
    os.makedirs("Plots", exist_ok=True)
    gif_path = "Plots/heat_diffusion_visualization.gif"
    gif_path_gradient = "Plots/heat_diffusion_visualization_gradient.gif"
    iio.imwrite(gif_path, frames, duration=100)  # 20 FPS
    iio.imwrite(gif_path_gradient, frames_gradient, duration=100)  # 20 FPS
    
    print(f"✅ GIF saved to {gif_path}")
    print(f"Visualization shows heat diffusion from a single hot point")
    print(f"Grid size: {H}x{W}, Time steps: {T}, Diffusion coefficient: {k}")