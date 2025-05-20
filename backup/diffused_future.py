import multiprocessing as mp
from multiprocessing import Process, Semaphore
import os
import numpy as np
import cv2
import matplotlib
import torch
from tqdm import tqdm

from event_tensor import build_event_tensor
from event_utils import load_events_fast, parse_meta
from heat import generate_heat_kernel_3d_np
from torch.utils.data import Dataset, DataLoader
from utils import make_side_by_side_video
from window_event import EventDataset
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Create the spawn context
# spawn_context = mp.get_context('spawn')
max_tasks = 4
# task_semaphore = Semaphore(max_tasks)

def run_one_diffusion(kernel_depth, sequence_dir, event_tensor_dirname, fps, k_index, mask_rgb_frames, device):
    k = np.logspace(np.log10(0.001), np.log10(100.0), num=6)[k_index]
    kernel = generate_heat_kernel_3d_np(kernel_depth, 33, 33, k=k)
    dataset = EventDataset(
        event_tensor_dir=os.path.join(sequence_dir, event_tensor_dirname),
        kernel_np=kernel,  # kernel gets converted to GPU inside Dataset
        rgb_frame_num=len(frame_info),
        height=260,
        width=346,
        # accum_frame=30,
        # diffuse_time=0.5
    )

    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
        # multiprocessing_context=spawn_context
    )

    H, W = 260, 346  # match dataset dimensions
    N = len(loader.dataset)
    video_out_path = f"videos/masked_event_diffusion_{k_index}.mp4" if mask_rgb_frames > 0 else f"videos/event_diffusion_{k_index}.mp4"

    # Preallocate video buffer
    # frame_buffer = np.zeros((N, H, W, 3), dtype=np.uint8)
    frame_buffer = np.zeros((N, H, W), dtype=np.uint16)

    # device = torch.device("cuda")
    
    # Assume frame_buffer is already allocated: (N, H, W, 3)
    for F_r_batch, diffused_batch in tqdm(loader, desc="Generating Event Diffusion Frames"): 
        diffused_batch = diffused_batch.to(device)  # shape: (B, H, W)
        for i, frame_index in enumerate(F_r_batch):
            frame_buffer[frame_index.item()] = diffused_batch[i].cpu().numpy()
    npy_filename = f"NPY/Masked_Diffused_Event_{k_idx}" if mask_rgb_frames > 0 else "NPY/Diffused_Event_{k_idx}"
    np.save(npy_filename, frame_buffer, allow_pickle=True)
    make_side_by_side_video(f"{sequence_dir}/img", event_tensor, frame_buffer, video_out_path, fps, 'mp4v')
    
def run_one_diffusion_wrapper(semaphore, gpu_id, *args):
    """
    Wrapper function to manage GPU semaphore and set device for run_one_diffusion.
    
    Parameters:
    - semaphore: Semaphore object to control GPU access.
    - gpu_id: Integer index of the GPU to use.
    - *args: Arguments to pass to run_one_diffusion.
    """
    semaphore.acquire()
    try:
        device = f'cuda:{gpu_id}'
        run_one_diffusion(*args, device)
    finally:
        semaphore.release()
        
if __name__ == "__main__":
    data_dir = "/storage/mostafizt/EVIMO/"
    object_name = "box"
    sequence_id = 11
    fps = 15
    k = np.logspace(np.log10(0.001), np.log10(100.0), num=6)[3] # Seems like k=3 provides best outcomes visually so I'm gonna use that for now
    event_step=1
    diffuse_time=2.0
    sequence_dir = f"{data_dir}/train/{object_name}/txt/seq_{sequence_id:02d}"
    meta_path = os.path.join(sequence_dir, 'meta.txt')
    event_path = os.path.join(sequence_dir, 'events.txt')
    mask_rgb_frames  = 0
    event_tensor_dirname = 'masked_event_tensor' if mask_rgb_frames >0 else 'event_tensor'
    os.makedirs('videos', exist_ok=True)
    os.makedirs("NPY", exist_ok=True)
    os.makedirs("Plots", exist_ok=True)
    frame_info = parse_meta(meta_path)
    all_events = load_events_fast(event_path)

    # Generate kernel on CPU as before
    # event_tensor, event_frame_rate, rgb_frame_rate, kernel_depth = build_event_tensor(events=all_events, frame_info=frame_info, height=260, width=346, event_step=event_step,
    #                                                                 diffuse_time=diffuse_time, mask_rgb_frames=mask_rgb_frames, device='cuda', dirname=os.path.join(sequence_dir, event_tensor_dirname))
    # print(f"Average Number of Events/RGB Frame: {event_frame_rate/rgb_frame_rate:.2f}")
    # # k_values np.logspace(np.log10(0.001), np.log10(3), num=6
    # for mask_rgb_frame in [0, 100]:
    #     for k_idx in [3]:
    #         # if not os.path.exists(f"videos/event_diffusion_{k_idx}.mp4"):
    #         print(f"Working with mask_rgb_frame : {mask_rgb_frame} and k_idx : {k_idx}")
    #         run_one_diffusion(kernel_depth, sequence_dir, event_tensor_dirname, fps, k_idx, mask_rgb_frame, 'cuda')
    # Detect available GPUs
    # available_gpus = list(range(torch.cuda.device_count()))
    # num_gpus = len(available_gpus)
    # print(f"🚀 Found {num_gpus} GPUs: {available_gpus}")

    # # Create one semaphore per GPU to limit access to one process at a time
    # gpu_semaphores = [Semaphore(1) for _ in range(num_gpus)]

    # processes = []
    # job_index = 0

    # # Distribute tasks across GPUs
    # for mask_rgb_frame in [0, 100]:
    #     for k_idx in range(5):
    #         print(f"Working with mask_rgb_frame : {mask_rgb_frame} and k_idx : {k_idx}")
    #         gpu_id = job_index % num_gpus
    #         p = Process(target=run_one_diffusion_wrapper, 
    #                     args=(gpu_semaphores[gpu_id], gpu_id, kernel_depth, sequence_dir, 
    #                           event_tensor_dirname, fps, k_idx, mask_rgb_frame))
    #         p.start()
    #         processes.append(p)
    #         job_index += 1

    # # Wait for all processes to complete
    # for p in processes:
    #     p.join()
    
    for mask_rgb_frames in [0, 100]:
        for k_idx in [3]:
            # ==== CONFIG ====
            file1 = f'NPY/Diffused_Event_{k_idx}.npy'
            file2 = f'NPY/Masked_Diffused_Event_{k_idx}.npy'
            output_video = f'videos/diff_video_{k_idx}.mp4'
            # ==== LOAD FILES ====
            arr1 = np.load(file1)  # shape: (628, H, W)
            arr2 = np.load(file2)

            assert arr1.shape == arr2.shape, "Shape mismatch!"

            # ==== ABS DIFFERENCE & MSE ====
            diff = np.abs(arr1 - arr2)
            mse_per_frame = np.mean((arr1[:-2] - arr2[:-2]) ** 2, axis=(1, 2))
            overall_mse = mse_per_frame.mean()
            print(f"✅ Overall MSE: {overall_mse:.4f}")

            # ==== PLOT MSE ====
            plt.figure(figsize=(10, 4))
            plt.plot(mse_per_frame, label="MSE per Frame")
            plt.xlabel("Frame Index")
            plt.ylabel("MSE")
            plt.title("Per-frame MSE")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"mse_plot_{k_idx}.png")
            plt.close()
            print("📊 MSE plot saved as mse_plot.png")

            # ==== NORMALIZATION ====
            diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)  # [0, 1]
            diff_norm_uint8 = (diff_norm * 255).astype(np.uint8)

            # ==== APPLY INFERNO COLORMAP ====
            colored_frames = [cv2.applyColorMap(f, cv2.COLORMAP_INFERNO) for f in diff_norm_uint8]

            # ==== SAVE VIDEO ====
            height, width, _ = colored_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            for frame in colored_frames:
                out.write(frame)
            out.release()
            print(f"🎞️ Diff video saved as {output_video}")


