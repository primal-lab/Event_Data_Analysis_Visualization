import os
from matplotlib import cm, pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm

_colormap_luts = {}  # global cache
lut = None

def torch_apply_colormap(batch: torch.Tensor, colormap_name: str = 'inferno') -> torch.Tensor:
    global _colormap_luts
    assert batch.ndim == 3, "Input must be a 3D tensor (B, H, W)"
    device = batch.device
    B, H, W = batch.shape

    # ✅ Build LUT once per device + colormap
    key = (colormap_name, device)
    if key not in _colormap_luts:
        cmap = plt.get_cmap(colormap_name)
        lut_np = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        _colormap_luts[key] = torch.tensor(lut_np, device=device, dtype=torch.uint8)

    lut = _colormap_luts[key]

    # Normalize batch per frame
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True)
    norm = batch / (mean + 5 * std + 1e-8)
    norm = norm.clamp(0, 1)
    indices = (norm * 255).long()

    flat_indices = indices.view(-1)
    flat_colors = lut[flat_indices]
    colored = flat_colors.view(B, H, W, 3)

    return colored

def apply_inferno_colormap(img):
    # Ensure img is 2D (grayscale)
    if img.ndim > 2 and img.shape[2] > 1:
        raise ValueError("Input image must be grayscale (single-channel)")
    elif img.ndim > 2:
        img = img.squeeze()

    # Normalize image to [0, 1] for colormap
    img_min, img_max = np.min(img), np.max(img)
    if img_min == img_max:
        img_normalized = np.zeros_like(img, dtype=np.float32)
    else:
        img_normalized = (img - img_min) / (img_max - img_min)
    
    # Apply inferno colormap
    inferno = cm.get_cmap('inferno')
    colored_img = inferno(img_normalized)[:, :, :3]  # Get RGB, discard alpha
    
    # Convert to uint8 for standard image format (0-255)
    colored_img = (colored_img * 255).astype(np.uint8)
    
    return colored_img


def make_side_by_side_video(img_path, img_list, event_tensor, heat_tensor, out_path, fps=30, codec="mp4v"):
    T = heat_tensor.shape[0]
    _, H, W = event_tensor.shape
    img_list = [os.path.join(img_path, i) for i in img_list]
    print(img_list)
    out_w, out_h = W * 3, H
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, int(fps), (out_w, out_h))
    # Normalization
    event_tensor = event_tensor.cpu().numpy()
    event_mean = np.mean(event_tensor)
    event_std = np.std(event_tensor)
    event = (255.*np.clip((event_tensor - 0)/(event_mean + 5*event_std + 1e-8), 0, 1)).astype(np.uint8)
    
    heat_mean = np.mean(heat_tensor)
    heat_std = np.std(heat_tensor)
    heat = (255.*np.clip((heat_tensor - 0)/(heat_mean + 5*heat_std + 1e-8), 0, 1)).astype(np.uint8)
    print(f"Event Tensor length: {len(event_tensor)} Diffused Tensor length: {len(heat_tensor)} Number of rgb frames : {len(img_list)}")
    for t in tqdm(range(T-1), desc="Generating video"): 
        if not os.path.exists(img_list[t]):
            continue
        img = cv2.imread(img_list[t])
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        
        # raw_col = cv2.applyColorMap(event[t], cv2.COLORMAP_INFERNO)

        raw_col = apply_inferno_colormap(event[t])
        
        raw_heat = cv2.applyColorMap(heat[t], cv2.COLORMAP_INFERNO)
        # print(f"Frame {t} Video heat sum {np.mean(raw_heat)}")
        combined = np.hstack([img, raw_col, raw_heat])
        writer.write(combined)

    writer.release()
    print(f" Video saved to {out_path}")