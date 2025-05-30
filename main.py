"""
Event Data Analysis and Visualization
Main execution script for event data processing and visualization.
"""

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_events_fast, parse_meta
from event_processing import build_event_tensor, EventDataset
from heat_analysis import generate_heat_kernel_3d_np
from heat_analysis.diffusion import generate_heat_kernel_gradient_3d_np
from visualization import make_side_by_side_video
from utils.directory_utils import setup_directories
from utils.frame_processing import process_frames, process_frames_cpu
from utils.data_utils import save_results
from config.config import (
    # Processing parameters
    KERNEL_SIZE,
    NUM_FRAMES,
    POLARITY_MODE,
    SEQUENCE_ID,
    EVENT_STEP,
    DIFFUSE_TIME,
    GRADIENT_PLOT,
    MASK_RGB_FRAMES,
    K,
    
    # Data dimensions
    HEIGHT,
    USE_CPU_PARALLEL,
    NUM_JOBS,
    WIDTH,
    
    # DataLoader settings
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
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
from visualization.quiver import generate_quiver_overlays

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main execution function."""
    # Setup
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f"Using device: {device}")
    
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
    event_tensor, event_frame_rate, rgb_frame_rate, kernel_depth, img_list, _ = build_event_tensor(
        events=all_events,
        frame_info=frame_info,
        height=HEIGHT,
        width=WIDTH,
        mode=POLARITY_MODE,
        num_rgb_frames=NUM_FRAMES,
        event_step=EVENT_STEP,
        diffuse_time=DIFFUSE_TIME,
        mask_rgb_frames=MASK_RGB_FRAMES,
        device=device,
        dirname=os.path.join(sequence_dir, event_tensor_dirname)
    )
    
    logger.info(f"Average Number of Events per RGB Frame: {event_frame_rate/rgb_frame_rate:.2f}")
    # Generate kernel
    idx = 0
    logger.info(f"Generating diffusion kernel and gradient for alpha: {K:.2e}")
    kernel = generate_heat_kernel_3d_np(kernel_depth, KERNEL_SIZE[0], KERNEL_SIZE[1], k=K)
    dH_dx_3d, dH_dy_3d = generate_heat_kernel_gradient_3d_np(kernel_depth, KERNEL_SIZE[0], KERNEL_SIZE[1], k=K)
    if USE_CPU_PARALLEL:
        logger.info("Processing frames using CPU parallel processing...")
        frame_buffer, frame_buffer_dx, frame_buffer_dy = process_frames_cpu(
            event_tensor_dir=os.path.join(sequence_dir, event_tensor_dirname),
            kernel=kernel,
            dH_dx_3d=dH_dx_3d,
            dH_dy_3d=dH_dy_3d,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            n_jobs=-1
        )
    else:
        # Original DataLoader processing
        logger.info("Processing frames using DataLoader...")
        dataset = EventDataset(
            event_tensor_dir=os.path.join(sequence_dir, event_tensor_dirname),
            kernel_np=kernel,
            rgb_frame_num=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH
        )
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR
        )
        
        frame_buffer = process_frames(loader, device)
    
    # Save results
    logger.info("Saving results...")
    save_results(frame_buffer, event_tensor, get_npy_filename(SEQUENCE_ID, idx))
    # Generate video
    logger.info("Generating visualization video...")
    if GRADIENT_PLOT:
        quiver_imgs = generate_quiver_overlays(frame_buffer_dx, frame_buffer_dy, img_list, f"{sequence_dir}/img", step=4)
    else:
        quiver_imgs = None
    make_side_by_side_video(
        f"{sequence_dir}/img",
        img_list,
        event_tensor,
        frame_buffer,
        get_video_out_path(SEQUENCE_ID, idx),
        FPS,
        VIDEO_CODEC,
        GRADIENT_PLOT,
        quiver_imgs
        
    )
    
    logger.info("Processing completed successfully!")

if __name__ == "__main__":
    main()