import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from event_processing.event_dataset import EventDataset
from visualization.video import generate_comparison_video, convert_to_heatmap
import cv2

def process_events_with_mask(mask_frames, data_dir, object_name, sequence_id, num_rgb_frames=520):
    """Process events with specified number of masked frames"""
    sequence_dir = f"{data_dir}/train/{object_name}/txt/seq_{sequence_id:02d}"
    event_tensor_dir = os.path.join(sequence_dir, "event_tensor")
    
    # Load kernel
    kernel_depth = 10
    kernel_size = 11
    sigma_t = 3
    sigma_s = 1.5
    
    # Create Gaussian kernel
    t = np.linspace(-kernel_depth//2, kernel_depth//2, kernel_depth)
    y = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    t, y, x = np.meshgrid(t, y, x, indexing='ij')
    
    kernel = np.exp(-(t**2/(2*sigma_t**2) + y**2/(2*sigma_s**2) + x**2/(2*sigma_s**2)))
    kernel = kernel / kernel.sum()
    
    # Create dataset
    dataset = EventDataset(
        event_tensor_dir=event_tensor_dir,
        kernel_np=kernel,
        rgb_frame_num=num_rgb_frames,
        height=346,
        width=260
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=24,
        pin_memory=True
    )
    
    # Process events
    diffused_events = []
    for F_r, diffused in tqdm(dataloader, desc=f"Processing events (mask={mask_frames})"):
        if F_r >= num_rgb_frames - mask_frames:
            diffused = torch.zeros_like(diffused)
        diffused_events.append(diffused.numpy().squeeze())
    
    return np.array(diffused_events)

def calculate_mse_and_create_diff_video(unmasked_data, masked_data, num_masked_frames, output_path):
    """Calculate MSE and create difference video for the masked frames"""
    # Calculate MSE for masked frames
    mse_values = []
    diff_frames = []
    
    for i in range(-num_masked_frames, 0):
        mse = np.mean((unmasked_data[i] - masked_data[i]) ** 2)
        mse_values.append(mse)
        
        # Create difference frame
        diff = np.abs(unmasked_data[i] - masked_data[i])
        diff_heatmap = convert_to_heatmap(diff)
        diff_frames.append(diff_heatmap)
    
    # Save MSE values
    mse_output = os.path.join(os.path.dirname(output_path), "mse_values.txt")
    with open(mse_output, "w") as f:
        for i, mse in enumerate(mse_values):
            f.write(f"Frame {num_masked_frames-i}: MSE = {mse}\n")
    
    # Create difference video
    H, W = diff_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    diff_video = cv2.VideoWriter(output_path, fourcc, 10, (W, H))
    
    for frame in diff_frames:
        diff_video.write(frame)
    
    diff_video.release()
    print(f"✅ Saved difference video: {output_path}")
    print(f"✅ Saved MSE values: {mse_output}")
    
    return mse_values

def main():
    # Parameters
    data_dir = "/storage/mostafizt/EVIMO"
    object_name = "box"
    sequence_id = 11
    num_rgb_frames = 520
    mask_frames_list = [0, 100]  # Process with 0 and 100 masked frames
    
    # Create output directory
    output_dir = "comparison_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process events with different masks in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for mask_frames in mask_frames_list:
            future = executor.submit(
                process_events_with_mask,
                mask_frames,
                data_dir,
                object_name,
                sequence_id,
                num_rgb_frames
            )
            futures.append((mask_frames, future))
        
        # Get results
        results = {}
        for mask_frames, future in futures:
            results[mask_frames] = future.result()
            # Save the diffused events
            output_path = os.path.join(output_dir, f"diffused_events_mask_{mask_frames}.npy")
            np.save(output_path, results[mask_frames])
            print(f"✅ Saved: {output_path}")
    
    # Generate comparison video
    generate_comparison_video(
        data_dir=data_dir,
        object_name=object_name,
        sequence_id=sequence_id,
        diffused_path=os.path.join(output_dir, "diffused_events_mask_0.npy"),
        masked_diffused_path=os.path.join(output_dir, "diffused_events_mask_100.npy"),
        output_path=os.path.join(output_dir, "comparison_video.mp4"),
        num_rgb_frames=num_rgb_frames,
        mask_frames=100,
        fps=10,
        resize_to=(260, 346)
    )
    
    # Calculate MSE and create difference video
    mse_values = calculate_mse_and_create_diff_video(
        results[0],  # unmasked
        results[100],  # masked
        100,  # num_masked_frames
        os.path.join(output_dir, "difference_video.mp4")
    )
    
    # Plot MSE values
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mse_values)), mse_values)
    plt.xlabel("Frame Index (from end)")
    plt.ylabel("MSE")
    plt.title("MSE between Masked and Unmasked Frames")
    plt.savefig(os.path.join(output_dir, "mse_plot.png"))
    plt.close()

if __name__ == "__main__":
    main() 