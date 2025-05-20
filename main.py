"""
Event Data Analysis and Visualization
Main execution script for event data processing and visualization.
"""

import os
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_events_fast, parse_meta
from event_processing import build_event_tensor, EventDataset
from processing import generate_heat_kernel_3d_np
from visualization import make_side_by_side_video
from config.config import (
    # Processing parameters
    EVENT_STEP,
    DIFFUSE_TIME,
    MASK_RGB_FRAMES,
    
    # Data dimensions
    HEIGHT,
    WIDTH,
    
    # DataLoader settings
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    
    # Video settings
    FPS,
    VIDEO_CODEC,
    
    # Path getters
    get_sequence_dir,
    get_meta_path, 
    get_event_path,
    get_event_tensor_dirname,
    get_video_out_path,
    get_npy_filename
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    dirs = ['NPY', 'Plots', 'videos']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def process_frames(loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Process frames using the dataloader and return the frame buffer.
    
    Args:
        loader (DataLoader): DataLoader containing the event data
        device (torch.device): Device to process on (CPU/GPU)
    
    Returns:
        np.ndarray: Processed frame buffer
    """
    H, W = HEIGHT, WIDTH
    N = len(loader.dataset)
    frame_buffer = np.zeros((N, H, W), dtype=np.float32)
    
    try:
        for F_r_batch, diffused_batch in tqdm(loader, desc="Generating Event Diffusion Frames"):
            diffused_batch = diffused_batch.to(device)
            for i, frame_index in enumerate(F_r_batch):
                frame_buffer[frame_index.item()] = diffused_batch[i].cpu().numpy()
    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        raise
    
    return frame_buffer

def save_results(frame_buffer: np.ndarray, event_tensor: torch.Tensor) -> None:
    """
    Save the processed results to NPY files.
    
    Args:
        frame_buffer (np.ndarray): Processed frame buffer
        event_tensor (torch.Tensor): Event tensor
    """
    try:
        npy_filename = get_npy_filename()
        np.save(npy_filename, frame_buffer, allow_pickle=True)
        np.save("NPY/Events", event_tensor.cpu().numpy(), allow_pickle=True)
        logger.info(f"Saved npy as {npy_filename}.npy")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main() -> None:
    """Main execution function."""
    try:
        # Setup
        setup_directories()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Setup paths
        sequence_dir = get_sequence_dir()
        meta_path = get_meta_path()
        event_path = get_event_path()
        event_tensor_dirname = get_event_tensor_dirname()
        
        # Load data
        logger.info("Loading event data and metadata...")
        frame_info = parse_meta(meta_path)
        all_events = load_events_fast(event_path)
        
        # Generate event tensor
        logger.info("Generating event tensor...")
        event_tensor, event_frame_rate, rgb_frame_rate, kernel_depth, img_list = build_event_tensor(
            events=all_events,
            frame_info=frame_info,
            height=HEIGHT,
            width=WIDTH,
            event_step=EVENT_STEP,
            diffuse_time=DIFFUSE_TIME,
            mask_rgb_frames=MASK_RGB_FRAMES,
            device=device,
            dirname=os.path.join(sequence_dir, event_tensor_dirname)
        )
        
        logger.info(f"Average Number of Events/RGB Frame: {event_frame_rate/rgb_frame_rate:.2f}")
        
        # Generate kernel
        logger.info("Generating diffusion kernel...")
        kernel = generate_heat_kernel_3d_np(kernel_depth, 33, 33, k=1.0)
        
        # Create dataset and dataloader
        logger.info("Setting up dataset and dataloader...")
        dataset = EventDataset(
            event_tensor_dir=os.path.join(sequence_dir, event_tensor_dirname),
            kernel_np=kernel,
            rgb_frame_num=520, # The video has 628 frames but after 520 frames, 
            #there seems to be some discontinuity in the rgb data so I hardcoded it to use 520 frames
            height=HEIGHT,
            width=WIDTH
        )
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS
        )
        
        # Process frames
        logger.info("Processing frames...")
        frame_buffer = process_frames(loader, device)
        
        # Save results
        logger.info("Saving results...")
        save_results(frame_buffer, event_tensor)
        
        # Generate video
        logger.info("Generating visualization video...")
        make_side_by_side_video(
            f"{sequence_dir}/img",
            img_list,
            event_tensor,
            frame_buffer,
            get_video_out_path(),
            FPS,
            VIDEO_CODEC
        )
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()