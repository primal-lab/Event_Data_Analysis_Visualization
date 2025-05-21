"""
Event Data Analysis and Visualization
Main execution script for event data processing and visualization.
"""

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from data import load_events_fast, parse_meta
from event_processing import build_event_tensor, EventDataset
from processing import generate_heat_kernel_3d_np
from visualization import make_side_by_side_video
from utils.directory_utils import setup_directories
from utils.frame_processing import process_frames, process_frames_cpu
from utils.data_utils import save_results
from config.config import (
    # Processing parameters
    EVENT_STEP,
    DIFFUSE_TIME,
    MASK_RGB_FRAMES,
    
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        event_tensor, event_frame_rate, rgb_frame_rate, kernel_depth, img_list, _ = build_event_tensor(
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
         
        if USE_CPU_PARALLEL:
            logger.info("Processing frames using CPU parallel processing...")
            frame_buffer = process_frames_cpu(
                event_tensor_dir=os.path.join(sequence_dir, event_tensor_dirname),
                kernel=kernel,
                num_frames=520,
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
                rgb_frame_num=520,
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
        save_results(frame_buffer, event_tensor, get_npy_filename())
        
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