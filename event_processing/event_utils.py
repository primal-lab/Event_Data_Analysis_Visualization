import os
import numpy as np
import torch

def load_events_fast(event_path):
    """
    Load events from a text file quickly using numpy.
    
    Args:
        event_path (str): Path to the events file
        
    Returns:
        dict: Dictionary containing event data (x, y, t, p)
    """
    # Load events using numpy's fast loading
    data = np.loadtxt(event_path, dtype=np.float32)
    data = data[np.isin(data[:, 3], [-1, 0, 1])]
    data[:, 1:4] = np.round(data[:, 1:4])
    
    # Convert to dictionary
    return {
        'x': torch.from_numpy(data[:, 0]),
        'y': torch.from_numpy(data[:, 1]),
        't': torch.from_numpy(data[:, 2]),
        'p': torch.from_numpy(data[:, 3])
    }

def parse_meta(meta_path):
    """
    Parse metadata file containing frame information.
    
    Args:
        meta_path (str): Path to the metadata file
        
    Returns:
        dict: Dictionary containing frame information
    """
    with open(meta_path, 'r') as f:
        data = eval(f.read())
    frames = data.get("frames", [])
    frame_info = sorted([(f['ts'], f['classical_frame']) for f in frames])
    
    # Convert to dictionary format
    frame_starts = np.array([f[0] for f in frame_info])
    frame_ends = np.array([f[0] for f in frame_info[1:]] + [frame_info[-1][0] + 0.1])  # Add small offset for last frame
    
    # Get image list
    img_dir = os.path.join(os.path.dirname(meta_path), 'img')
    img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    return {
        'frame_starts': frame_starts,
        'frame_ends': frame_ends,
        'img_list': img_list
    }

def load_events_fast(event_path):
    data = np.loadtxt(event_path, dtype=np.float32)
    data = data[np.isin(data[:, 3], [-1, 0, 1])]
    data[:, 1:4] = np.round(data[:, 1:4])
    return data 