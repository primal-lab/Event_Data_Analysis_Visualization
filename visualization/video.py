"""
Video generation and processing utilities.
Contains functions for creating side-by-side comparison videos.
"""

from matplotlib import cm
import numpy as np
import cv2
import os
from tqdm import tqdm

from data import parse_meta

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

def convert_to_heatmap(diffused_frame):
    mean = diffused_frame.mean()
    std = diffused_frame.std()
    normalized = np.clip((diffused_frame) / (mean + 4 * std + 1e-8), 0, 1) # For some reason, the heatmap for img normalized with mean + 4*std looks better
    img = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    return heatmap

def generate_comparison_video(
    data_dir, object_name, sequence_id,
    diffused_path, masked_diffused_path,
    output_path="heatmap_comparison_grid.mp4",
    num_rgb_frames=520, mask_frames=100, fps=10, resize_to=(320, 320)
):
    sequence_dir = f"{data_dir}/train/{object_name}/txt/seq_{sequence_id:02d}"
    img_path = os.path.join(sequence_dir, "img")
    meta_path = os.path.join(sequence_dir, "meta.txt")

    frame_info = parse_meta(meta_path)[:num_rgb_frames]
    img_list = [os.path.join(img_path, fname) for (_, fname) in frame_info]

    print("Loading RGB frames...")
    event_data = np.array([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in tqdm(img_list)])
    masked_rgb = event_data.copy()
    masked_rgb[-mask_frames:] = 0

    print("Loading heatmaps...")
    diffused_data = np.load(diffused_path)[:num_rgb_frames]
    masked_diffused_data = np.load(masked_diffused_path)[:num_rgb_frames]

    H, W = resize_to
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (2 * W, 2 * H))

    print("Generating video...")
    for i in tqdm(range(num_rgb_frames)):
        orig_rgb = cv2.resize(event_data[i], (W, H))
        orig_rgb_color = cv2.cvtColor(orig_rgb, cv2.COLOR_GRAY2BGR)

        orig_heatmap = convert_to_heatmap(cv2.resize(diffused_data[i], (W, H)))
        masked_rgb_frame = cv2.resize(masked_rgb[i], (W, H))
        masked_rgb_color = cv2.cvtColor(masked_rgb_frame, cv2.COLOR_GRAY2BGR)

        masked_heatmap = convert_to_heatmap(cv2.resize(masked_diffused_data[i], (W, H)))

        top_row = np.hstack((orig_rgb_color, orig_heatmap))
        bottom_row = np.hstack((masked_rgb_color, masked_heatmap))
        grid_frame = np.vstack((top_row, bottom_row))

        out.write(grid_frame)

    out.release()
    print(f"✅ Saved: {output_path}")


def make_side_by_side_video(img_path, img_list, event_tensor, heat_tensor, out_path, fps=30, codec="mp4v"):
    T = heat_tensor.shape[0]
    _, H, W = event_tensor.shape
    img_list = [os.path.join(img_path, i) for i in img_list]
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

if __name__ == "__main__":
    generate_comparison_video("/storage/mostafizt/EVIMO/", "box", 11, "NPY/Diffused_Event_3.npy", "NPY/Masked_Diffused_Event_3.npy", resize_to=(260, 346))
