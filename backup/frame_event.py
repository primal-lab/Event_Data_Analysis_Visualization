import os
import numpy as np
import torch
from torch.utils.data import Dataset

def process_frame(i, events_cpu, frame_starts, event_step):
    # mask = events_cpu["frame_idx"] == i
    frame_start = frame_starts[i]
    mask = events_cpu["t"] < frame_start

    if not np.any(mask):
        return  # skip empty frames

    x_f = events_cpu["x"][mask]
    y_f = events_cpu["y"][mask]
    t_f = events_cpu["t"][mask]
    p_f = events_cpu["p"][mask]

    unique_ts = np.unique(t_f)
    sampled_ts = unique_ts[::event_step]
    keep_mask = np.isin(t_f, sampled_ts)

    x_f = x_f[keep_mask]
    y_f = y_f[keep_mask]
    t_f = t_f[keep_mask]
    p_f = p_f[keep_mask]
    # for t in t_f:
    #     print(frame_starts -t)

    if len(x_f) == 0:
        return  # skip saving empty file

    event_data = torch.tensor(np.stack([x_f, y_f, p_f, t_f], axis=1), dtype=torch.float64)
    frame_info_dict = {"event_data": event_data, "frame_start": frame_starts[i]}
    torch.save(frame_info_dict, f"event_tensor/frame_{i:04d}.pt")
    

class EventDataset(Dataset):
    def __init__(self, event_tensor_dir, kernel_np, rgb_frame_num, height, width, accum_frame, diffuse_time=1.0):
        """
        Args:
            event_tensor_dir (str): Path to folder containing frame_{i:04d}.pt event files
            kernel_np (np.ndarray): Precomputed 3D kernel, shape (K, kH, kW)
            rgb_frame_num (int): Total number of RGB frames
            height (int): Height of the target diffusion map
            width (int): Width of the target diffusion map
            accum_frame (int): Number of RGB frames to look back for accumulation
        """
        self.event_tensor_dir = event_tensor_dir
        self.kernel_np = kernel_np  # will be converted to torch.Tensor on first use
        self.rgb_frame_num = rgb_frame_num
        self.height = height
        self.width = width
        self.accum_frame = accum_frame
        self.diffuse_time = diffuse_time

        self.kernel_depth, self.kH, self.kW = kernel_np.shape
        self.pad_h = self.kH // 2
        self.pad_w = self.kW // 2
        self.device = torch.device("cuda")

    def __len__(self):
        return self.rgb_frame_num

    def __getitem__(self, F_r):
        """
        CPU-only version of event accumulation.
        Returns raw float32 numpy slice for post-processing on GPU.
        """
        slice_accum = np.zeros((self.height + 2 * self.pad_h,
                                self.width + 2 * self.pad_w),
                            dtype=np.float32)

        if not isinstance(self.kernel_np, np.ndarray):
            raise ValueError("kernel_np must be a NumPy array in CPU mode.")

        final_frame_start = None
        accumulated_events = []
        earliest_time_seen = None

        for f in range(F_r, -1, -1):
            frame_path = os.path.join(self.event_tensor_dir, f"frame_{f:04d}.pt")
            if not os.path.exists(frame_path):
                continue

            data = torch.load(frame_path, weights_only=False)
            events = data["event_data"]
            frame_start = data["frame_start"]

            if final_frame_start is None:
                final_frame_start = frame_start

            if events.numel() == 0:
                continue

            min_event_time = events[:, 3].min().item()
            if earliest_time_seen is None or min_event_time < earliest_time_seen:
                earliest_time_seen = min_event_time

            accumulated_events.append(events)

            if final_frame_start - earliest_time_seen >= self.diffuse_time:
                break

        if len(accumulated_events) == 0:
            return F_r, slice_accum[self.pad_h:self.pad_h+self.height, self.pad_w:self.pad_w+self.width]

        all_events = torch.cat(accumulated_events, dim=0).numpy()

        for x, y, p, t in all_events:
            delta_t = final_frame_start - t
            delta_t_idx = int(delta_t*self.kernel_depth/self.diffuse_time)

            if delta_t_idx >= self.kernel_depth or delta_t_idx < 0:
                continue

            x, y = int(x), int(y)
            x_p, y_p = x + self.pad_w, y + self.pad_h

            if (x_p - self.pad_w < 0 or x_p + self.pad_w + 1 > self.width + 2 * self.pad_w or
                y_p - self.pad_h < 0 or y_p + self.pad_h + 1 > self.height + 2 * self.pad_h):
                continue

            slice_accum[y_p - self.pad_h : y_p + self.pad_h + 1,
                        x_p - self.pad_w : x_p + self.pad_w + 1] += self.kernel_np[delta_t_idx]*p

        cropped = slice_accum[self.pad_h:self.pad_h + self.height,
                            self.pad_w:self.pad_w + self.width]
        return F_r, cropped

