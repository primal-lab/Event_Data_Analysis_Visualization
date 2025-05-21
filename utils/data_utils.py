"""
Utility functions for data saving and result handling.
"""
import numpy as np
import torch

def save_results(frame_buffer: np.ndarray, event_tensor: torch.Tensor, npy_filename: str) -> None:
    """
    Save the processed results to NPY files.
    
    Args:
        frame_buffer (np.ndarray): Processed frame buffer
        event_tensor (torch.Tensor): Event tensor
        npy_filename (str): Path to save the frame buffer
    """
    try:
        np.save(npy_filename, frame_buffer, allow_pickle=True)
        np.save("NPY/Events", event_tensor.cpu().numpy(), allow_pickle=True)
        print(f"✅ Saved npy as {npy_filename}.npy")
    except Exception as e:
        raise Exception(f"Error saving results: {str(e)}") 