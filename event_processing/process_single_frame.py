import os
import numpy as np
import torch

def process_single_frame(args):
    """
    Process a single frame of event data with diffusion kernel.
    
    Args:
        args (tuple): Contains:
            - F_r (int): Frame index
            - event_tensor_dir (str): Directory containing event window files
            - kernel_np (np.ndarray): 3D diffusion kernel
            - height (int): Height of output frame
            - width (int): Width of output frame
            
    Returns:
        tuple: (frame_index, diffused_frame)
    """
    F_r, event_tensor_dir, kernel_np, height, width = args
    
    # Get kernel dimensions
    kernel_depth, kH, kW = kernel_np.shape
    pad_h = kH // 2
    pad_w = kW // 2
    
    # Allocate padded canvas
    slice_accum = np.zeros((height + 2 * pad_h, width + 2 * pad_w), dtype=np.float32)
    
    # Load event data
    window_path = os.path.join(event_tensor_dir, f"window_{F_r:04d}.pt")
    if not os.path.exists(window_path):
        # Return blank if no event data
        cropped = slice_accum[pad_h:pad_h + height, pad_w:pad_w + width]
        return F_r, cropped
        
    # Load and process events
    events = torch.load(window_path)
    events_np = events.numpy()
    
    # Extract columns
    x, y, p, delta_t_idx = events_np.T
    x = x.astype(int)
    y = y.astype(int)
    delta_t_idx = delta_t_idx.astype(int)
    
    # Compute padded coordinates
    x_p = x + pad_w
    y_p = y + pad_h
    
    # Precompute slice indices
    y_start = y_p - pad_h
    y_end = y_p + pad_h + 1
    x_start = x_p - pad_w
    x_end = x_p + pad_w + 1
    
    # Accumulate contributions
    for i in range(len(x_p)):
        slice_accum[y_start[i]:y_end[i], x_start[i]:x_end[i]] += kernel_np[delta_t_idx[i]] * p[i]
    
    # Crop padding
    cropped = slice_accum[pad_h:pad_h + height, pad_w:pad_w + width]
    
    return F_r, cropped 