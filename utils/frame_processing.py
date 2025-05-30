"""
Utility functions for frame processing operations.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from tqdm import tqdm

from event_processing.process_single_frame import process_single_frame

def process_frames(loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Process frames using the dataloader and return the frame buffer.
    
    Args:
        loader (DataLoader): DataLoader containing the event data
        device (torch.device): Device to process on (CPU/GPU)
    
    Returns:
        np.ndarray: Processed frame buffer
    """
    H, W = loader.dataset.height, loader.dataset.width
    N = len(loader.dataset)
    frame_buffer = np.zeros((N, H, W), dtype=np.float32)
    
    try:
        for F_r_batch, diffused_batch in tqdm(loader, desc="Generating Event Diffusion Frames"):
            diffused_batch = diffused_batch.to(device)
            # Process each frame in the batch
            for i, frame_index in enumerate(F_r_batch):
                frame_buffer[frame_index.item()] = diffused_batch[i].cpu().numpy()
    except Exception as e:
        raise Exception(f"Error processing frames: {str(e)}")
    
    return frame_buffer

def process_frames_cpu(event_tensor_dir: str, kernel: np.ndarray, dH_dx_3d: np.ndarray, dH_dy_3d: np.ndarray, num_frames: int, height: int, width: int, n_jobs: int = -1) -> np.ndarray:
    """
    Process frames using joblib parallel processing on CPU.
    
    Args:
        event_tensor_dir (str): Directory containing event tensor files
        kernel (np.ndarray): 3D diffusion kernel
        num_frames (int): Number of frames to process
        height (int): Height of output frame
        width (int): Width of output frame
        n_jobs (int): Number of parallel jobs (-1 for all cores)
    
    Returns:
        np.ndarray: Processed frame buffer
    """
    frame_buffer = np.zeros((num_frames, height, width), dtype=np.float32)
    frame_buffer_dx = np.zeros((num_frames, height, width), dtype=np.float32)
    frame_buffer_dy = np.zeros((num_frames, height, width), dtype=np.float32)
    
    # Prepare arguments for parallel processing
    args_list = [(i, event_tensor_dir, kernel, dH_dx_3d, dH_dy_3d, height, width) for i in range(num_frames)]
    
    # Process frames in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_frame)(args) for args in tqdm(args_list, desc="Processing frames", colour='green')
    )
    
    # Fill frame buffer with results
    for frame_idx, frame_data, frame_data_dx, frame_data_dy in results:
        frame_buffer[frame_idx] = frame_data
        frame_buffer_dx[frame_idx] = frame_data_dx
        frame_buffer_dy[frame_idx] = frame_data_dy
        
    return frame_buffer, frame_buffer_dx, frame_buffer_dy 