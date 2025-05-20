"""
Not used in the current implementation.
"""
import numpy as np
import torch

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