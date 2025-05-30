"""
Configuration parameters for Event Data Analysis and Visualization.
All parameters are organized into logical sections for better maintainability.
"""

import os

# ================ Data Paths ================
DATA_DIR = "/storage/mostafizt/EVIMO/"
OBJECT_NAME = "box"
SEQUENCE_ID = 1

# ================ Event Processing Parameters ================
EVENT_STEP = 1
DIFFUSE_TIME = 0.75
MASK_RGB_FRAMES = 0

# ================ Image Dimensions ================
HEIGHT = 260
WIDTH = 346

# ================ Diffusion Parameters ================
K = 0.2  # Alpha value for heat kernel
KERNEL_SIZE = (51, 51)  # (height, width) of the kernel

GRADIENT_PLOT = True
# ================ CPU Parallel Processing Parameters ================
USE_CPU_PARALLEL = True
NUM_JOBS = -1

# ================ DataLoader Parameters ================
BATCH_SIZE = 1 # Batch size is 1 is typically faster than batch size > 1
    # because some event data has a lot of events and it slows down processing of the whole batch
NUM_FRAMES = 800
NUM_WORKERS = 24
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
# ================ Video Parameters ================
FPS = 15
VIDEO_CODEC = 'mp4v'

# ================ Derived Paths ================
def get_sequence_dir():
    """Get the sequence directory path."""
    return os.path.join(DATA_DIR, "train", OBJECT_NAME, "txt", f"seq_{SEQUENCE_ID:02d}")

def get_meta_path():
    """Get the metadata file path."""
    return os.path.join(get_sequence_dir(), 'meta.txt')

def get_event_path():
    """Get the events file path."""
    return os.path.join(get_sequence_dir(), 'events.txt')

def get_event_tensor_dirname():
    """Get the event tensor directory name based on masking."""
    return 'masked_event_tensor' if MASK_RGB_FRAMES > 0 else 'event_tensor'

def get_video_out_path(sequence_id, k_idx):
    """Get the output video path."""
    base_name = f"masked_event_diffusion_{sequence_id}_{k_idx}.mp4" if MASK_RGB_FRAMES > 0 else f"event_diffusion_{sequence_id}_{k_idx}.mp4"
    return os.path.join("videos", base_name)

def get_npy_filename(sequence_id, k_idx):
    """Get the output NPY filename."""
    base_name = f"Masked_Diffused_Event_{sequence_id}_{k_idx}" if MASK_RGB_FRAMES > 0 else f"Diffused_Event_{sequence_id}_{k_idx}" 
    return os.path.join("NPY", base_name) 