import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
import os
from tqdm import tqdm

def generate_quiver_overlays(dH_dx_3d, dH_dy_3d, img_list, img_path,
                              scale=1.0, step=10, max_magnitude=5.0,
                              save_debug_png=False):
    """
    Generate quiver overlays with arrow lengths proportional to magnitude (clipped).

    Parameters:
        dH_dx_3d (ndarray): Gradient in x direction, shape (T, H, W)
        dH_dy_3d (ndarray): Gradient in y direction, shape (T, H, W)
        img_list (list): List of image filenames
        img_path (str): Base path to load images from
        scale (float): Visual scale factor for arrow length
        step (int): Sampling stride for quiver arrows
        max_magnitude (float): Maximum magnitude for arrow clipping
        save_debug_png (bool): Whether to save each frame as PNG for inspection

    Returns:
        List of overlay images as RGB numpy arrays
    """
    overlays = []
    T, H, W = dH_dx_3d.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    for t in tqdm(range(T), desc="Generating quiver overlays"):
        # Load and resize base image
        img = cv2.imread(os.path.join(img_path, img_list[t]))
        img = cv2.resize(img, (W, H))

        # Extract gradients
        U_full = -dH_dx_3d[t]
        V_full = -dH_dy_3d[t]

        # Downsample
        U = U_full[::step, ::step]
        V = V_full[::step, ::step]
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]

        # Compute magnitude and clip
        mags = np.sqrt(U**2 + V**2)
        mags_clipped = np.clip(mags, 0, max_magnitude)

        # Normalize vectors but scale by clipped magnitude
        U_scaled = (U / (mags.sum() + 1e-8)) * 1 * scale
        V_scaled = (V / (mags.sum() + 1e-8)) * 1 * scale

        # Create figure
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img)
        ax.quiver(Xs, Ys, U_scaled, V_scaled, mags_clipped,
                  cmap='inferno', scale=None, width=0.004, headwidth=3)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if save_debug_png:
            plt.savefig(f'gradient_{t:03d}.png', bbox_inches='tight', pad_inches=0)

        # Render to image
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        overlay_img = np.array(Image.open(buf))
        overlays.append(overlay_img[:, :, :3])
        plt.close(fig)

    return overlays
