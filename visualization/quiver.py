import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import numpy as np
from tqdm import tqdm

def generate_quiver_overlays(dH_dx_3d, dH_dy_3d, img_list, img_path, scale=1.0):
    """
    Generate quiver overlays and return a list of RGB images with quiver arrows.

    Parameters:
        dH_dx_3d (ndarray): Gradient in x direction, shape (T, H, W)
        dH_dy_3d (ndarray): Gradient in y direction, shape (T, H, W)
        img_list (list): List of image filenames
        img_path (str): Base path to load images from
        scale (float): Arrow scaling factor

    Returns:
        List of quiver overlay images as np.ndarray (same shape as RGB frame)
    """
    overlays = []
    T, H, W = dH_dx_3d.shape

    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    for t in tqdm(range(T), desc="Generating quiver overlays"):
        # Load base image
        img = cv2.imread(os.path.join(img_path, img_list[t]))
        img = cv2.resize(img, (W, H))

        # Create a figure without borders
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img)
        ax.quiver(X, Y, dH_dx_3d[t], dH_dy_3d[t], 
                  color='cyan', angles='xy', scale_units='xy', scale=scale, width=0.001)

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Render figure to numpy image
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        overlay_img = np.array(Image.open(buf))
        overlays.append(overlay_img[:, :, :3])  # Remove alpha channel if present
        plt.close(fig)

    return overlays

