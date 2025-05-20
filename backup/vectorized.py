import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

from tqdm import tqdm

from event_utils import parse_meta

def compute_sobel_gradients(frame):
    dH_dx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    dH_dy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    return dH_dx, dH_dy

from scipy.signal import convolve2d

def precompute_heat_kernel_and_derivatives(size=31, alpha=1.0, t=1.0):
    """
    Precompute heat kernel and its spatial derivatives.
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
    """
    grad_x = convolve2d(frame, dH_dx, mode='same', boundary='symm')
    grad_y = convolve2d(frame, dH_dy, mode='same', boundary='symm')
    return grad_x, grad_y

def generate_quiver_overlay_cv2(img, dH_dx, dH_dy, step=10, scale=2.5):
    """
    Generate overlay image with arrows using OpenCV (much faster).
    event_frame: (H, W), grayscale
    dH_dx, dH_dy: gradient maps
    Returns: BGR image with arrows
    """
    H, W, _ = img.shape
    # overlay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # overlay = cv2.cvtColor((255 * (img / img.max())).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for y in range(0, H, step):
        for x in range(0, W, step):
            dx = dH_dx[y, x]
            dy = dH_dy[y, x]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-3:
                end_x = int(x + scale * dx)
                end_y = int(y + scale * dy)
                cv2.arrowedLine(img, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)
    return img


def create_quiver_video(event_data_path, diffused_data_path, save_path="quiver_overlay_sobel.mp4",
                        step=12, scale=1.5, fps=30):
    data_dir = "/storage/mostafizt/EVIMO/"
    object_name = "box"
    sequence_id = 11
    k = np.logspace(np.log10(0.001), np.log10(100.0), num=6)[3]
    event_step=1
    diffuse_time=2.0
    # Load data
    event_data = np.load(event_data_path)
    diffused_data = np.load(diffused_data_path)
    sequence_dir = f"{data_dir}/train/{object_name}/txt/seq_{sequence_id:02d}"
    img_path = f"{sequence_dir}/img/"
    meta_path = os.path.join(sequence_dir, 'meta.txt')
    frame_info = parse_meta(meta_path)
    num_rgb_frames = 520
    frame_starts = np.array([t for (t, _) in frame_info])[:520] # Hardcoded 520 because there is a gap in the event data after that
    img_list = np.array([img for (_, img) in frame_info])[:520]
    img_list = [os.path.join(img_path, i) for i in img_list]
    assert event_data.shape == diffused_data.shape, "Shape mismatch between event and diffused data"
    T, H, W = event_data.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for i in tqdm(range(T)):
        img = cv2.imread(img_list[i])
        dH_dx, dH_dy = compute_sobel_gradients(diffused_data[i])
        # print(img.shape,  diffused_data[i].shape)
        left_overlay = generate_quiver_overlay_cv2(img, dH_dx, dH_dy, step=step, scale=scale)
        d = diffused_data[i]
        d_norm = np.clip((d) / (np.mean(d) + 4 * np.std(d) + 1e-8), 0, 1)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)

        # Overlay arrows
        right_overlay = generate_quiver_overlay_cv2(heatmap, dH_dx, dH_dy, step=step, scale=scale)

        # --- Combine side by side ---
        combined = np.hstack((left_overlay, right_overlay))
        cv2.imwrite(f"vector{i}.png", combined)
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✅ Video saved to: {save_path}")

# Example usage
create_quiver_video("NPY/Events.npy", "NPY/Diffused_Event_3.npy", step=20, fps=15)
