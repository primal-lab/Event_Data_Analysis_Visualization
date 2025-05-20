import os
from joblib import Parallel, delayed
import numpy as np
import torch
from tqdm import tqdm

from frame_event import process_frame
from window_event import process_window

def build_event_tensor(events, frame_info, height, width,  event_step=10, diffuse_time=0.5, mask_rgb_frames= 0, device='cuda', dirname = "/storage/mostafizt/EVIMO/train/box/txt/seq_11/event_tensor"):
    """
    Builds a (num_rgb_frames, H, W) event tensor using scatter_add.

    Args:
        events (np.ndarray): shape (N, 3 or 4) = [timestamp, x, y, (optional polarity)]
        frame_info (list of tuples): each tuple = (start_time, ...), len = num_frames + 1
        height, width (int): spatial dimensions
        event_step (int): optional subsampling rate
        device (str): target device

    Returns:
        event_tensor (torch.Tensor): shape (num_rgb_frames, height, width)
    """
    print("⚡ Building event tensor on GPU...")

    # num_rgb_frames = len(frame_info) - 1
    num_rgb_frames = 520
    frame_starts = np.array([t for (t, _) in frame_info])[:520] # Hardcoded 520 because there is a gap in the event data after that
    img_list = np.array([img for (_, img) in frame_info])[:520]
    # print(f"Legit Image File length : {(img_list)}")
    event_times_np = events[:, 0]

    # Map each event to its corresponding RGB frame
    frame_indices = np.searchsorted(frame_starts, event_times_np, side="right") - 1
    valid_rgb = (frame_indices >= 0) & (frame_indices < num_rgb_frames)
    frame_indices = frame_indices[valid_rgb]
    events = events[valid_rgb]
    event_times_np = event_times_np[valid_rgb]
    event_frame_rate = len(np.unique(event_times_np))/(np.max(event_times_np) - np.min(event_times_np))
    rgb_frame_rate = num_rgb_frames/(np.max(frame_starts) - np.min(frame_starts))
    print(f"Event Frame Rate: {event_frame_rate:.2f}")
    print(f"RGB Frame Rate: {rgb_frame_rate:.2f}")
    kernel_depth = int(event_frame_rate*diffuse_time/event_step)
    # Convert to torch
    frame_indices_t = torch.tensor(frame_indices, dtype=torch.long, device=device)
    x = torch.tensor(events[:, 1], dtype=torch.long, device=device)
    y = torch.tensor(events[:, 2], dtype=torch.long, device=device)
    ts = torch.tensor(event_times_np, dtype=torch.float32, device=device)

    if events.shape[1] > 3:
        polarity = torch.tensor(events[:, 3], dtype=torch.float32, device=device)
        # polarity[polarity == 0] = -1  # ✅ map 0 to -1
    else:
        polarity = torch.zeros_like(x, dtype=torch.float32)  
    # Filter spatially valid
    valid_pos = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid_pos]
    y = y[valid_pos]
    ts = ts[valid_pos]
    polarity = polarity[valid_pos]
    frame_indices_t = frame_indices_t[valid_pos]

    # Allocate output
    event_tensor = torch.zeros((num_rgb_frames, height, width), dtype=torch.float32, device=device)

    # Flattened indexing into (F, H, W)
    flat_idx = frame_indices_t * height * width + y * width + x
    # ones = polarity  # weighted accumulation, or use torch.ones_like(flat_idx) for binary
    ones = torch.ones_like(flat_idx, dtype=torch.float32)
    event_tensor.view(-1).scatter_add_(0, flat_idx, ones)
    os.makedirs(dirname, exist_ok=True)

    events_cpu = {
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "t": ts.detach().cpu().numpy(),
        "p": (polarity.detach().cpu().numpy() - 0.0)*1,
        "frame_idx": frame_indices_t.detach().cpu().numpy()
    }
    
    # Masking condition
    t_cutoff = num_rgb_frames - mask_rgb_frames
    events_cpu["p"][events_cpu["frame_idx"] >= t_cutoff] = 0
    print("💽 Saving per-frame event tensors to 'event_tensor/frame_*.pt' ...")
    num_jobs = -1  # use all cores, or set to e.g. 8

    # Parallel(n_jobs=num_jobs)(
    #     delayed(process_frame)(i, events_cpu, frame_starts, event_step)
    #     for i in tqdm(range(num_rgb_frames), desc="💾 Saving event frames")
    # )
    
    Parallel(n_jobs=num_jobs)(
    delayed(process_window)(
        i,
        events_cpu,
        frame_starts,
        kernel_depth=kernel_depth,
        diffuse_time=diffuse_time,
        event_step=event_step,
        height=height,
        width=width,
        dirname = dirname
    )
    for i in tqdm(range(num_rgb_frames), desc="💾 Saving diffusion windows")
)


    return event_tensor, event_frame_rate, rgb_frame_rate, kernel_depth, img_list
