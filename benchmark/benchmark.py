import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse
import os
from numba import njit, prange
import multiprocessing

@njit
def accumulate_heat_kernel(y_p, x_p, p, heat_kernel, slice_accum, pad):
    kH, kW = heat_kernel.shape
    for i in range(len(x_p)):
        y_start = y_p[i] - pad
        y_end = y_p[i] + pad + 1
        x_start = x_p[i] - pad
        x_end = x_p[i] + pad + 1
        y_slice_start = max(0, y_start)
        y_slice_end = min(slice_accum.shape[0], y_end)
        x_slice_start = max(0, x_start)
        x_slice_end = min(slice_accum.shape[1], x_end)
        for y_idx in range(y_slice_start, y_slice_end):
            for x_idx in range(x_slice_start, x_slice_end):
                ky = y_idx - y_start
                kx = x_idx - x_start
                slice_accum[y_idx, x_idx] += heat_kernel[ky, kx] * p[i]

@njit(parallel=True)
def accumulate_heat_kernel_parallel_batches(y_p, x_p, p, heat_kernel, slice_accums, batch_indices, pad):
    for batch_idx in prange(len(batch_indices) - 1):
        start_idx = batch_indices[batch_idx]
        end_idx = batch_indices[batch_idx + 1]
        batch_y_p = y_p[start_idx:end_idx]
        batch_x_p = x_p[start_idx:end_idx]
        batch_p = p[start_idx:end_idx]
        accumulate_heat_kernel(batch_y_p, batch_x_p, batch_p, heat_kernel, slice_accums[batch_idx], pad)

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Gaussian kernel functions
def gaussian_kernel_2d(kernel_size, sigma, device):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (4 * sigma + 1e-8))
    kernel = kernel / kernel.sum()
    assert abs(kernel.sum() - 1.0) < 1e-10, "2D kernel not properly normalized"
    return kernel

def gaussian_kernel_1d(kernel_size, sigma, device):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=torch.float64)
    kernel = torch.exp(-(ax**2) / (4 * sigma + 1e-8))
    kernel = kernel / kernel.sum()
    assert abs(kernel.sum() - 1.0) < 1e-10, "1D kernel not properly normalized"
    return kernel

# Benchmark parameters
num_points_values = [1] + list(np.logspace(np.log10(100), np.log10(500000), 200).astype(int))
methods = ['No Precomputation', 'Heat Kernel Precomputed', 'Scipy Implementation', 'Torch Conv2D', 'Torch Separable Convolution']
results = []
num_batches = max(1, multiprocessing.cpu_count())
height, width = 1024, 1024
sigma = 10.0
kernel_size = int(6 * sigma + 1)
if kernel_size % 2 == 0:
    kernel_size += 1
pad = kernel_size // 2

os.makedirs("precomputed_outputs", exist_ok=True)

# Precompute heat kernel once
dx = np.arange(-pad, pad + 1)
dy = np.arange(-pad, pad + 1)
X, Y = np.meshgrid(dx, dy, indexing='ij')
heat_kernel = np.exp(-(X**2 + Y**2) / (4 * sigma + 1e-8))
heat_kernel /= heat_kernel.sum()
assert abs(heat_kernel.sum() - 1.0) < 1e-10, "Kernel not normalized"
heat_kernel = heat_kernel.astype(np.float64)

for num_points in tqdm(num_points_values):
    np.random.seed(42)
    if num_points == 1:
        events_np = np.array([[50.0, 50.0, 1.0, 0.0]])
    else:
        events_np = np.random.rand(num_points, 4) * min(height, width)

    # Original (dynamic kernel placement using heat equation)
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float64)
    start_time = time.time()
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        x_p, y_p = x + pad, y + pad
        slice_accum[y_p - pad:y_p + pad + 1, x_p - pad:x_p + pad + 1] += heat_kernel * p
    original_cropped = slice_accum[pad:-pad, pad:-pad]
    original_time = time.time() - start_time

    # Precomputed (Numba-optimized with parallel batches)
    x, y, p, _ = events_np.T
    x_p = np.array([int(xi % width) + pad for xi in x], dtype=np.int32)
    y_p = np.array([int(yi % height) + pad for yi in y], dtype=np.int32)
    p = p.astype(np.float64)

    start_time = time.time()
    effective_batches = min(num_batches, num_points)
    batch_size = max(1, num_points // effective_batches)
    batch_indices = [i * batch_size for i in range(effective_batches)]
    batch_indices.append(num_points)
    slice_accums = np.zeros((effective_batches, height + 2 * pad, width + 2 * pad), dtype=np.float64)
    accumulate_heat_kernel_parallel_batches(y_p, x_p, p, heat_kernel, slice_accums, batch_indices, pad)
    slice_accum = np.sum(slice_accums, axis=0)
    precomputed_output = slice_accum[pad:-pad, pad:-pad]
    precomputed_time = time.time() - start_time
    precomputed_mse = mse(original_cropped, precomputed_output)

    plt.imsave(f"precomputed_outputs/precomputed_{num_points}.png", precomputed_output, cmap='hot')

    # Scipy Implementation (using convolve2d)
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float64)
    start_time = time.time()
    event_map = np.zeros((height, width), dtype=np.float64)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p
    blurred = convolve2d(event_map, heat_kernel, mode='full')
    slice_accum = blurred[pad:-pad, pad:-pad]
    scipy_output = slice_accum
    scipy_time = time.time() - start_time
    scipy_mse = mse(original_cropped, scipy_output)

    # Torch Conv2D
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float64)
    start_time = time.time()
    event_map = np.zeros((height, width), dtype=np.float64)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p
    event_map_tensor = torch.from_numpy(event_map).to(device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernel_2d(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(event_map_tensor, kernel, padding=pad)
    slice_accum[pad:-pad, pad:-pad] = blurred.squeeze().cpu().numpy()
    torch_output = slice_accum[pad:-pad, pad:-pad]
    torch_time = time.time() - start_time
    torch_mse = mse(original_cropped, torch_output)

    # Torch Separable Convolution
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float64)
    start_time = time.time()
    event_map = np.zeros((height, width), dtype=np.float64)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p
    event_map_tensor = torch.from_numpy(event_map).to(device, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    k1d = gaussian_kernel_1d(kernel_size, sigma, device)
    kernel_h = k1d.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    kernel_v = k1d.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    temp = F.conv2d(event_map_tensor, kernel_h, padding=(pad, 0))
    blurred = F.conv2d(temp, kernel_v, padding=(0, pad))
    slice_accum[pad:-pad, pad:-pad] = blurred.squeeze().cpu().numpy()
    sep_output = slice_accum[pad:-pad, pad:-pad]
    sep_time = time.time() - start_time
    sep_mse = mse(original_cropped, sep_output)

    # Store results
    results.extend([
        ['No Precomputation', num_points, original_time, 0.0, 1.0],
        ['Heat Kernel Precomputed', num_points, precomputed_time, precomputed_mse, original_time / precomputed_time],
        ['Scipy Implementation', num_points, scipy_time, scipy_mse, original_time / scipy_time],
        ['Torch Conv2D', num_points, torch_time, torch_mse, original_time / torch_time],
        ['Torch Separable Convolution', num_points, sep_time, sep_mse, original_time / sep_time]
    ])

# Output results table
import pandas as pd
results_df = pd.DataFrame(results, columns=['Method', 'Num_Points', 'Runtime(s)', 'MSE', 'Speedup'])
print(results_df)

# Plotting
plt.figure(figsize=(16, 8))
for method in methods:
    subset = results_df[results_df['Method'] == method]
    if subset.empty:
        print(f"⚠️ Skipping method {method} due to missing data")
        continue
    plt.plot(subset['Num_Points'], subset['Runtime(s)'], label=method)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('Number of Points')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs Number of Points')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('num_points_vs_runtime.png')
plt.close()

plt.figure(figsize=(16, 8))
for method in methods[1:]:
    subset = results_df[results_df['Method'] == method]
    if subset.empty:
        print(f"⚠️ Skipping method {method} in MSE plot")
        continue
    plt.plot(subset['Num_Points'], subset['MSE'], label=method)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('Number of Points')
plt.ylabel('MSE')
plt.title('MSE vs Number of Points')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('num_points_vs_mse.png')
plt.close()

plt.figure(figsize=(16, 8))
for method in methods[1:]:
    subset = results_df[results_df['Method'] == method]
    if subset.empty:
        print(f"⚠️ Skipping method {method} in Speedup plot")
        continue
    plt.plot(subset['Num_Points'], subset['Speedup'], label=method)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('Number of Points')
plt.ylabel('SpeedUp')
plt.title('SpeedUp vs Number of Points')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('num_points_vs_speedup.png')
plt.close()