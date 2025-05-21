# Add these at the top of event_processing/event_dataset.py, after the existing imports
import concurrent.futures
import os
from queue import Queue

import numpy as np
import torch

from event_processing.event_dataset import EventDataset

def adaptive_collate_fn(batch):
    """
    Custom collate function that handles mixed batch sizes.
    Place this function before the class definitions.
    """
    # Separate frames that need individual processing
    individual_frames = []
    batch_frames = []
    
    for F_r, data, needs_individual in batch:
        # Convert numpy array to torch tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        if needs_individual:
            individual_frames.append((F_r, data))
        else:
            batch_frames.append((F_r, data))
    
    # Process and return both types
    results = []
    
    # Process individual frames
    for F_r, data in individual_frames:
        results.append((F_r, data))
    
    # Batch process the rest if any exist
    if batch_frames:
        F_rs, datas = zip(*batch_frames)
        # Ensure all tensors are on the same device
        datas = [d.to(datas[0].device) for d in datas]
        results.extend(zip(F_rs, torch.stack(datas)))
    
    return results

# Add this class after your existing EventDataset class
class AsyncAdaptiveEventDataset(EventDataset):
    def __init__(self, event_tensor_dir, kernel_np, rgb_frame_num, height, width, 
                 event_threshold=10000, prefetch_size=3):
        """
        Async and adaptive version of EventDataset.
        
        Additional Args:
            event_threshold (int): Number of events above which a frame is processed individually
            prefetch_size (int): Number of frames to prefetch
        """
        super().__init__(event_tensor_dir, kernel_np, rgb_frame_num, height, width)
        self.event_threshold = event_threshold
        self.prefetch_size = prefetch_size
        
        # Create thread pool for async loading
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_size)
        self.futures = {}
        
        # Start prefetching first few frames
        self._prefetch_frames(0, prefetch_size)

    def _load_frame(self, F_r):
        """Load a single frame asynchronously"""
        window_path = os.path.join(self.event_tensor_dir, f"window_{F_r:04d}.pt")
        try:
            if not os.path.exists(window_path):
                return F_r, None
            events = torch.load(window_path)
            return F_r, events
        except Exception as e:
            print(f"Error loading frame {F_r}: {e}")
            return F_r, None

    def _prefetch_frames(self, start_idx, num_frames):
        """Submit multiple frames for prefetching"""
        for i in range(start_idx, min(start_idx + num_frames, self.rgb_frame_num)):
            if i not in self.futures:
                self.futures[i] = self.pool.submit(self._load_frame, i)

    def __getitem__(self, F_r):
        """
        Async and adaptive version of event processing.
        Returns:
            F_r (int): Frame index
            cropped (torch.Tensor): Diffused event heatmap
            needs_individual (bool): Whether this frame should be processed individually
        """
        # Start loading next few frames
        self._prefetch_frames(F_r + 1, self.prefetch_size)
        
        # Get current frame
        if F_r in self.futures:
            frame_idx, events = self.futures[F_r].result()
            del self.futures[F_r]
        else:
            frame_idx, events = self._load_frame(F_r)

        # Initialize accumulator
        slice_accum = np.zeros((self.height + 2 * self.pad_h,
                              self.width + 2 * self.pad_w), dtype=np.float32)

        # Handle empty or missing data
        if events is None or len(events) == 0:
            cropped = slice_accum[self.pad_h:self.pad_h + self.height,
                                self.pad_w:self.pad_w + self.width]
            # Convert to torch tensor before returning
            return F_r, torch.from_numpy(cropped), False

        # Check if this frame needs individual processing
        needs_individual = len(events) > self.event_threshold

        # Process events using parent class logic
        events_np = events.numpy()
        x, y, p, delta_t_idx = events_np.T
        x = x.astype(int)
        y = y.astype(int)
        delta_t_idx = delta_t_idx.astype(int)

        # Compute padded coordinates
        x_p = x + self.pad_w
        y_p = y + self.pad_h

        # Precompute slice indices
        y_start = y_p - self.pad_h
        y_end = y_p + self.pad_h + 1
        x_start = x_p - self.pad_w
        x_end = x_p + self.pad_w + 1

        # Accumulate contributions
        for i in range(len(x_p)):
            slice_accum[y_start[i]:y_end[i], x_start[i]:x_end[i]] += self.kernel_np[delta_t_idx[i]] * p[i]

        # Crop padding
        cropped = slice_accum[self.pad_h:self.pad_h + self.height,
                            self.pad_w:self.pad_w + self.width]

        # Convert to torch tensor before returning
        return F_r, torch.from_numpy(cropped), needs_individual

    def __del__(self):
        """Cleanup thread pool on deletion"""
        self.pool.shutdown(wait=False)