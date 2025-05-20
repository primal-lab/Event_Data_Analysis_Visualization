"""
Configuration parameters for Event Data Analysis and Visualization.
All parameters are organized into logical sections for better maintainability.
"""

import os

# ================ Data Paths ================
DATA_DIR = "/storage/mostafizt/EVIMO/"
OBJECT_NAME = "box"
SEQUENCE_ID = 11

# ================ Event Processing Parameters ================
EVENT_STEP = 1
DIFFUSE_TIME = 2.0
MASK_RGB_FRAMES = 100

# ================ Image Dimensions ================
HEIGHT = 260
WIDTH = 346

# ================ Diffusion Parameters ================
K = 1.0  # Alpha value for heat kernel
KERNEL_SIZE = (33, 33)  # (height, width) of the kernel

# ================ DataLoader Parameters ================
BATCH_SIZE = 20
NUM_WORKERS = 24
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ================ Video Parameters ================
FPS = 10
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

def get_video_out_path():
    """Get the output video path."""
    base_name = "masked_event_diffusion_3.mp4" if MASK_RGB_FRAMES > 0 else "event_diffusion_3.mp4"
    return os.path.join("videos", base_name)

def get_npy_filename():
    """Get the output NPY filename."""
    base_name = "Masked_Diffused_Event_3" if MASK_RGB_FRAMES > 0 else "Diffused_Event_3"
    return os.path.join("NPY", base_name) 