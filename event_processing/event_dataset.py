import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EventDataset(Dataset):
    def __init__(self, event_tensor_dir, kernel_np, rgb_frame_num, height, width):
        """
        Dataset for applying heat diffusion to precomputed per-frame event windows.

        Each frame's file should be named window_{F_r:04d}.pt and contain:
            [x, y, polarity, delta_t_idx] for all events within the diffuse window

        Args:
            event_tensor_dir (str): Path to folder containing window_XXXX.pt files
            kernel_np (np.ndarray): 3D diffusion kernel of shape (T, kH, kW)
            rgb_frame_num (int): Number of RGB frames (total frames)
            height, width (int): Dimensions of the output diffusion maps
        """
        self.event_tensor_dir = event_tensor_dir
        self.kernel_np = kernel_np  # expected as NumPy array
        self.rgb_frame_num = rgb_frame_num
        self.height = height
        self.width = width

        self.kernel_depth, self.kH, self.kW = kernel_np.shape
        self.pad_h = self.kH // 2
        self.pad_w = self.kW // 2

    def __len__(self):
        return self.rgb_frame_num

    def __getitem__(self, F_r):
        """
        For frame index F_r, loads the precomputed event window file and applies diffusion.

        Returns:
            F_r (int): Frame index
            cropped (np.ndarray): Diffused event heatmap of shape (H, W), dtype float32
        """
        # Allocate padded canvas for accumulating diffused values
        slice_accum = np.zeros((self.height + 2 * self.pad_h,
                                self.width + 2 * self.pad_w), dtype=np.float32)

        # Construct event window file path
        window_path = os.path.join(self.event_tensor_dir, f"window_{F_r:04d}.pt")
        if not os.path.exists(window_path):
            # Return blank if no event data
            cropped = slice_accum[self.pad_h:self.pad_h + self.height,
                                  self.pad_w:self.pad_w + self.width]
            return F_r, cropped

        # Load precomputed event data (x, y, p, delta_t_idx)
        events = torch.load(window_path)  # shape: (N, 4)
        events_np = events.numpy()
        # if F_r>=628-60:print(f"Frame: {F_r} event summation: {np.sum(events_np)}")
        # Extract columns from events_np (assumed pre-filtered)
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

        # for x, y, p, delta_t_idx in events_np:
        #     delta_t_idx = int(delta_t_idx)
        #     if delta_t_idx < 0 or delta_t_idx >= self.kernel_depth:
        #         continue  # skip invalid indices

        #     x, y = int(x), int(y)
        #     x_p, y_p = x + self.pad_w, y + self.pad_h

        #     # Boundary check
        #     if (x_p - self.pad_w < 0 or x_p + self.pad_w + 1 > self.width + 2 * self.pad_w or
        #         y_p - self.pad_h < 0 or y_p + self.pad_h + 1 > self.height + 2 * self.pad_h):
        #         continue

        #     # Apply kernel weighted by polarity/intensity
        #     slice_accum[y_p - self.pad_h : y_p + self.pad_h + 1,
        #                 x_p - self.pad_w : x_p + self.pad_w + 1] += self.kernel_np[delta_t_idx] * p

        # Crop padding to get final (H, W) result
        cropped = slice_accum[self.pad_h:self.pad_h + self.height,
                              self.pad_w:self.pad_w + self.width]
        # if F_r >= 500:
        #     print(f"Frame: {F_r} Sum: {np.sum(cropped)}")
        return F_r, cropped