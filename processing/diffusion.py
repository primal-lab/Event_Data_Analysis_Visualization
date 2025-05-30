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
    exponent = -(x**2 + y**2) / (4 * k * t + 1e-8)
    H =  (1 / denom) * np.exp(exponent)
    return H/H.sum()


def heat_kernel_gradient(x, y, t, k):
    eps = 1e-8
    H = heat_kernel(x, y, t, k)

    dH_dx = H * (-x / (2 * k * (t + eps)))
    dH_dy = H * (-y / (2 * k * (t + eps)))
    # print(f"x: {x}, y: {y} t: {t} H_max : {H[x, y]} Alpha: {k:.2e} Extra Grad: {-x/ (2 * k * (t + eps))} Max dH_dx: {dH_dx[x, y]}")
    return dH_dx, dH_dy

def generate_heat_kernel_3d_np(T, H, W, k=0.05):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel(X, Y, t/T, k)
        # if t%300==0: print(f"At time t: {t/T:.2e} Kenel Sum: {np.sum(kernel[t - 1])} and max: {kernel[t - 1].max()}")
    return kernel

def generate_heat_kernel_gradient_3d_np(T, H, W, k=0.05):
    
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    dH_dx_3d = np.zeros((T, H, W), dtype=np.float32)
    dH_dy_3d = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        dH_dx, dH_dy = heat_kernel_gradient(X, Y, t/T, k)
        dH_dx_3d[t - 1] = dH_dx.astype(np.float32)
        dH_dy_3d[t - 1] = dH_dy.astype(np.float32)
    
    return dH_dx_3d, dH_dy_3d

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