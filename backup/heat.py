from matplotlib import pyplot as plt
import numpy as np
import matplotlib
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
        kernel[t - 1] = heat_kernel(X, Y, t/T, k)
        kernel[t - 1] /= kernel[t - 1].sum()
    return kernel

if __name__ == "__main__":
    # --- Generate Kernel Sequence ---
    T, H, W = 5000, 33, 33
    kernel = generate_heat_kernel_3d_np(T, H, W, k=0.05)

    # --- Create GIF Frames as Heatmaps ---
    frames = []
    for t in range(0, T, 50):  # every 250th frame
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(kernel[t], cmap='inferno', interpolation='nearest', vmax=0.5)
        ax.axis('off')  # hide axes
        fig.tight_layout(pad=0)

        # Convert to image buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # --- Save GIF ---
    gif_path = "heat_kernel_confusionmap.gif"
    iio.imwrite(gif_path, frames, duration=50)  # ~20 FPS

    print(f"✅ GIF saved to {gif_path}")