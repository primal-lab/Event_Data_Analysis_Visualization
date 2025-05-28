import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Heat kernel and gradient
def heat_kernel(rel_x, rel_y, t, k, eps=1e-8):
    t += eps
    denom = 4 * np.pi * k * t
    exp_term = np.exp(-(rel_x**2 + rel_y**2) / (4 * k * t))
    kernel = exp_term / denom
    return kernel / (np.sum(kernel) + eps)

def heat_grad(rel_x, rel_y, t, k, eps=1e-8):
    kernel = heat_kernel(rel_x, rel_y, t, k, eps)
    grad = -kernel / (2 * k * t + eps)
    return grad * rel_x , grad * rel_y

# Compute heat and flux
def compute_heat_flux(X, Y, points, t, k, L, size):
    if t == 0:
        Z = np.zeros_like(X)
        x_idx = ((points[:, 0] + L/2) * (size - 1) / L).astype(int)
        y_idx = ((points[:, 1] + L/2) * (size - 1) / L).astype(int)
        for i, c in enumerate(points[:, 2]):
            if 0 <= x_idx[i] < size and 0 <= y_idx[i] < size:
                Z[y_idx[i], x_idx[i]] = c
        return Z, np.zeros_like(X), np.zeros_like(Y)

    Z = np.zeros_like(X)
    gx, gy = np.zeros_like(X), np.zeros_like(Y)
    for x0, y0, c in points:
        rel_x = X - x0
        rel_y = Y - y0
        kernel = heat_kernel(rel_x, rel_y, t, k)
        grad_x, grad_y = heat_grad(rel_x, rel_y, t, k)
        Z += c * kernel
        gx += c * grad_x
        gy += c * grad_y
    return Z, -k * gx, -k * gy

# Generate single frame
def generate_frame(i, t, points, output, vmax, X, Y, L, k, size):
    Z, qx, qy = compute_heat_flux(X, Y, points, t, k, L, size)
    fig = Figure(figsize=(12, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(1, 2)

    im = ax1.imshow(Z.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2],
                    cmap='inferno', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax1, label='Temperature')
    ax1.set(title=f"t={t:.4f}s", xlabel='X', ylabel='Y', aspect='equal')

    skip = max(1, size // 48)
    # skip = 1
    qscale = max(np.sqrt(qx**2 + qy**2).max(), 1e-8) * 5
    ax2.quiver(X[::skip, ::skip].T, Y[::skip, ::skip].T,
               qx[::skip, ::skip].T, qy[::skip, ::skip].T,
               scale=qscale, color='red', alpha=0.8) # scale=qscale
    ax2.set(title=f"Flux t={t:.4f}s", xlabel='X', ylabel='Y',
            aspect='equal', xlim=[-L/2, L/2], ylim=[-L/2, L/2])

    fname = f"{output}/frame_{i:04d}.png"
    canvas.print_png(fname)
    plt.close(fig)
    return fname

# Main script
def main():
    size, L, k = 101, 1.0, 0.003
    num_points, C = 1, 1.0
    start_t, end_t, dt, fps = 0.0001, 0.50, 0.001, 10
    output = "heat_diffusion_frames"
    os.makedirs(output, exist_ok=True)

    np.random.seed(42)
    # pos = L * (np.random.rand(num_points, 2) - 0.5)*0
    pos = np.array([[0.0    , -0.3]])
    points = np.column_stack([pos, np.full(num_points, C)])

    coords = np.linspace(-L/2, L/2, size)
    X, Y = np.meshgrid(coords, coords, indexing='xy')
    times = np.arange(start_t, end_t + dt, dt)

    Z0, _, _ = compute_heat_flux(X, Y, points, times[0], k, L, size)
    vmax = 0.3

    frame_args = [(i, t, points, output, vmax, X, Y, L, k, size)
                  for i, t in enumerate(times)]

    print(f"Generating {len(times)} frames...")
    frames = Parallel(n_jobs=-1)(delayed(generate_frame)(*args) for args in tqdm(frame_args))

    with imageio.get_writer("heat_diffusion_2d.gif", mode='I', fps=fps) as writer:
        for f in frames:
            writer.append_data(imageio.imread(f))
            os.remove(f)
    os.rmdir(output)
    print("GIF complete!")

if __name__ == "__main__":
    main()
