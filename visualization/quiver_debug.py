import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio
import os
from joblib import Parallel, delayed
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Spatial domain
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
X_exp, Y_exp = X[..., np.newaxis], Y[..., np.newaxis]

# Parameters
k = 2  # Diffusion coefficient
start_t, end_t, dt = 0.001, 3, 0.03
t_specific = 1.7  # Time for SSIM comparison and plots
step = 2  # Adjusted downsampling for quiver plot

# Directories
output_dir = 'frames'
magnitude_dir = 'gradient_magnitudes'
plot_dir = 'plots_t_1.7'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(magnitude_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Heat kernel function
def compute_heat_kernel(t, k, X, Y, center_points):
    c = 1 / (4 * np.pi * k * t)
    Z = c * np.exp(-((X_exp - center_points[:, 0])**2 + (Y_exp - center_points[:, 1])**2) * np.pi * c)
    return np.sum(Z, axis=-1)

# Gradient kernel function
def compute_heat_grad(t, k, X, Y, center_points):
    c = 1 / (4 * np.pi * k * t)
    Z = c * np.exp(-((X_exp - center_points[:, 0])**2 + (Y_exp - center_points[:, 1])**2) * np.pi * c)
    dZ_dx = -2 * (X_exp - center_points[:, 0]) * Z
    dZ_dy = -2 * (Y_exp - center_points[:, 1]) * Z
    dZ_dx_total = np.sum(dZ_dx, axis=-1)
    dZ_dy_total = np.sum(dZ_dy, axis=-1)
    return dZ_dx_total, dZ_dy_total

# Add circles around center points
def add_circles(ax, centers, radius=0.1, color='red'):
    for center in centers:
        circle = Circle(center, radius, color=color, fill=False, linewidth=2)
        ax.add_patch(circle)

# Generate a single frame or plot
def generate_frame(t, center_points, frame_dir=None, save_frame=False, plot_filename=None):
    Z_total = compute_heat_kernel(t, k, X, Y, center_points)
    dZ_dx_total, dZ_dy_total = compute_heat_grad(t, k, X, Y, center_points)
    magnitude = np.sqrt(dZ_dx_total**2 + dZ_dy_total**2)
    
    # Debug: Check gradient magnitudes at t=2.1
    if np.isclose(t, t_specific, atol=dt/2):
        print(f"N={len(center_points)}, t={t:.3f}, dZ_dx_total min/max: {dZ_dx_total.min():.4e}/{dZ_dx_total.max():.4e}")
        print(f"N={len(center_points)}, t={t:.3f}, dZ_dy_total min/max: {dZ_dy_total.min():.4e}/{dZ_dy_total.max():.4e}")
        print(f"N={len(center_points)}, t={t:.3f}, magnitude min/max: {magnitude.min():.4e}/{magnitude.max():.4e}")
    
    if save_frame or plot_filename:
        Xs, Ys = X[::step, ::step], Y[::step, ::step]
        U, V = -dZ_dx_total[::step, ::step], -dZ_dy_total[::step, ::step]
        
        # Normalize quiver arrows
        mag = np.sqrt(U**2 + V**2)
        mag[mag == 0] = 1e-8  # Avoid division by zero
        U_norm, V_norm = U / mag, V / mag
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))
        
        # Scalar field
        im = ax1.imshow(Z_total, extent=(-2, 2, -2, 2), origin='lower', cmap='inferno', vmin=0, vmax=0.4)
        ax1.set_title(f'Scalar Field, t={t:.3f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        add_circles(ax1, center_points)
        
        # Quiver plot (normalized arrows)
        ax2.quiver(Xs, Ys, U_norm, V_norm, angles='xy', scale_units='xy', scale=5, width=0.004)
        ax2.set_title('Direction of Steepest Decrease (Normalized)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        add_circles(ax2, center_points)
        
        # Gradient magnitude
        mag_im = ax3.imshow(magnitude, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
        ax3.set_title('Gradient Magnitude')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        fig.colorbar(mag_im, ax=ax3, fraction=0.046, pad=0.04)
        add_circles(ax3, center_points)
        
        plt.tight_layout()
        
        if save_frame and frame_dir:
            frame_filename = os.path.join(frame_dir, f'frame_{int(t*1000):04d}.png')
            plt.savefig(frame_filename)
        elif plot_filename:
            plt.savefig(plot_filename)
        
        plt.close(fig)
    
    return magnitude if np.isclose(t, t_specific, atol=dt/2) else None

# Create GIF for N=14
def create_gif(N, output_gif):
    np.random.seed(42)
    center_points = np.random.uniform(-2, 2, (N, 2))
    print(f"N={N}, Center Points:\n{center_points}")
    frame_dir = os.path.join(output_dir, f'N_{N}')
    os.makedirs(frame_dir, exist_ok=True)
    
    times = np.arange(start_t, end_t, dt)
    magnitudes = Parallel(n_jobs=-1)(
        delayed(generate_frame)(t, center_points, frame_dir=frame_dir, save_frame=True) 
        for t in tqdm(times, desc=f"Generating frames for N={N}")
    )
    
    # Create GIF
    frame_paths = []
    for t in times:
        frame_path = os.path.join(frame_dir, f'frame_{int(t*1000):04d}.png')
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    
    if frame_paths:
        frames = [imageio.imread(frame_path) for frame_path in frame_paths]
        imageio.mimsave(output_gif, frames, fps=10)
    
    # Clean up frames
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            os.remove(frame_path)
    if os.path.exists(frame_dir) and not os.listdir(frame_dir):
        os.rmdir(frame_dir)
    
    return [m for m in magnitudes if m is not None][0]  # Return magnitude at t=2.1

# Process for N=5 to 30
ssim_values = []
base_magnitude = None

for N in tqdm(range(5, 31), desc="Processing N"):
    np.random.seed(42)
    center_points = np.random.uniform(-2, 2, (N, 2))
    
    # Generate GIF for N=14
    if N == 14:
        magnitude = create_gif(N, 'heat_dissipation_N14.gif')
    else:
        # Compute magnitude at t=2.1 without saving frames
        magnitude = generate_frame(t_specific, center_points, plot_filename=None, save_frame=False)
    
    # Save plot for t=2.1
    plot_filename = os.path.join(plot_dir, f'plot_N{N}_t2.1.png')
    generate_frame(t_specific, center_points, plot_filename=plot_filename, save_frame=False)
    
    # Save gradient magnitude
    np.save(os.path.join(magnitude_dir, f'magnitude_N{N}.npy'), magnitude)
    
    # Compute SSIM
    if N == 5:
        base_magnitude = magnitude
    else:
        ssim_value = ssim(base_magnitude, magnitude, data_range=magnitude.max() - magnitude.min())
        ssim_values.append((N, ssim_value))

# Print SSIM values
for N, ssim_value in ssim_values:
    print(f'SSIM for N={N} vs N=5 at t=2.1: {ssim_value:.4f}')

# Clean up directories
if os.path.exists(output_dir) and not os.listdir(output_dir):
    os.rmdir(output_dir)
if os.path.exists(magnitude_dir) and not os.listdir(magnitude_dir):
    os.rmdir(magnitude_dir)
if os.path.exists(plot_dir) and not os.listdir(plot_dir):
    os.rmdir(plot_dir)