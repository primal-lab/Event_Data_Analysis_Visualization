import numpy as np
import cv2
import os
from tqdm import tqdm

from event_utils import parse_meta


def convert_to_heatmap(diffused_frame):
    mean = diffused_frame.mean()
    std = diffused_frame.std()
    normalized = np.clip((diffused_frame) / (mean + 4 * std + 1e-8), 0, 1)
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

generate_comparison_video("/storage/mostafizt/EVIMO/", "box", 11, "NPY/Diffused_Event_3.npy", "NPY/Masked_Diffused_Event_3.npy", resize_to=(260, 346))