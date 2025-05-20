import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.signal import convolve2d

# === Diffusion Functions ===
def original_method(events_np, height, width, pad, sigma):
    """Apply diffusion using dynamic kernel placement with heat equation."""
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float64)
    start_time = time.time()
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        x_p, y_p = x + pad, y + pad
        dx = np.arange(-pad, pad + 1)
        dy = np.arange(-pad, pad + 1)
        X, Y = np.meshgrid(dx, dy, indexing='ij')
        heat_kernel = np.exp(-(X**2 + Y**2) / (4 * sigma + 1e-8))
        heat_kernel /= heat_kernel.sum()
        slice_accum[y_p - pad:y_p + pad + 1, x_p - pad:x_p + pad + 1] += heat_kernel * p
    cropped = slice_accum[pad:-pad, pad:-pad]
    time_taken = time.time() - start_time
    return cropped, time_taken

def precomputed_method(events_np, height, width, pad, precomputed_kernel):
    import time
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float32)
    start_time = time.time()

    event_map = np.zeros((height, width), dtype=np.float32)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p

    blurred = convolve2d(event_map, precomputed_kernel, mode='full')
    slice_accum[pad:-pad, pad:-pad] = blurred[pad:-pad, pad:-pad]
    cropped = slice_accum[pad:-pad, pad:-pad]
    time_taken = time.time() - start_time
    return cropped, time_taken

def scipy_gaussian_method(events_np, height, width, pad, sigma, device, kernel_size):
    import time
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float32)
    start_time = time.time()

    event_map = np.zeros((height, width), dtype=np.float32)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p

    event_map_tensor = torch.tensor(event_map, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernel_2d(kernel_size, sigma, device).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    blurred = F.conv2d(event_map_tensor, kernel, padding=pad)
    slice_accum[pad:-pad, pad:-pad] = blurred.squeeze().cpu().numpy()
    cropped = slice_accum[pad:-pad, pad:-pad]
    time_taken = time.time() - start_time
    return cropped, time_taken


def dense_torch_method(events_np, height, width, pad, sigma, device, kernel_size):
    import time
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float32)
    start_time = time.time()

    event_map = np.zeros((height, width), dtype=np.float32)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p

    event_map_tensor = torch.tensor(event_map, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernel_2d(kernel_size, sigma, device).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    blurred = F.conv2d(event_map_tensor, kernel, padding=pad)
    slice_accum[pad:-pad, pad:-pad] = blurred.squeeze().cpu().numpy()
    cropped = slice_accum[pad:-pad, pad:-pad]
    time_taken = time.time() - start_time
    return cropped, time_taken

def separable_torch_method(events_np, height, width, pad, sigma, device, kernel_size):
    import time
    slice_accum = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.float32)
    start_time = time.time()

    event_map = np.zeros((height, width), dtype=np.float32)
    for x, y, p, _ in events_np:
        x, y = int(x % width), int(y % height)
        event_map[y, x] += p

    event_map_tensor = torch.tensor(event_map, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    k1d = gaussian_kernel_1d(kernel_size, sigma, device).to(dtype=torch.float32)
    kernel_h = k1d.view(1, 1, -1, 1)
    kernel_v = k1d.view(1, 1, 1, -1)

    temp = F.conv2d(event_map_tensor, kernel_h, padding=(pad, 0))
    blurred = F.conv2d(temp, kernel_v, padding=(0, pad))
    slice_accum[pad:-pad, pad:-pad] = blurred.squeeze().cpu().numpy()
    cropped = slice_accum[pad:-pad, pad:-pad]
    time_taken = time.time() - start_time
    return cropped, time_taken

# === Helper Functions ===
def gaussian_kernel_2d(kernel_size, sigma, device):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

def gaussian_kernel_1d(kernel_size, sigma, device):
    """Generate a 1D Gaussian kernel for separable convolution."""
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
    kernel = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).view(1, 1, -1)

def mse(output1, output2):
    """Compute Mean Squared Error between two arrays."""
    return np.mean((output1 - output2)**2)

# === Event Processing Class ===
class EventProcessor:
    def __init__(self, height, width, pad, sigma, device, kernel_size, method='original'):
        self.height = height
        self.width = width
        self.pad = pad
        self.sigma = sigma
        self.device = device
        self.kernel_size = kernel_size
        self.method = method
        # Precompute the kernel only once
        dx = np.arange(-pad, pad + 1)
        dy = np.arange(-pad, pad + 1)
        X, Y = np.meshgrid(dx, dy, indexing='ij')
        self.precomputed_kernel = np.exp(-(X**2 + Y**2) / (4 * sigma + 1e-8)).astype(np.float32)
        self.precomputed_kernel /= self.precomputed_kernel.sum()

    def process_events(self, events_np):
        """Process events using the specified diffusion method."""
        if self.method == 'original':
            output, time_taken = original_method(events_np, self.height, self.width, self.pad, self.sigma)
        elif self.method == 'precomputed':
            output, time_taken = precomputed_method(events_np, self.height, self.width, self.pad, self.precomputed_kernel)
        elif self.method == 'scipy_gaussian':
            output, time_taken = scipy_gaussian_method(events_np, self.height, self.width, self.pad, self.sigma, self.device, self.kernel_size)
        elif self.method == 'dense_torch':
            output, time_taken = dense_torch_method(events_np, self.height, self.width, self.pad, self.sigma, self.device, self.kernel_size)
        elif self.method == 'separable_torch':
            output, time_taken = separable_torch_method(events_np, self.height, self.width, self.pad, self.sigma, self.device, self.kernel_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return output, time_taken

# === Example Usage ===
if __name__ == "__main__":
    # Settings
    height, width = 260, 346
    pad = 16
    sigma = 2.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kernel_size = 33
    num_points_values = [100, 600, 1000, 5000]

    for num_points in num_points_values:
        np.random.seed(42)
        events_np = np.random.rand(num_points, 4) * min(height, width)
    
        processor = EventProcessor(height, width, pad, sigma, device, kernel_size, method='original')
        original_output, original_time = processor.process_events(events_np)

        processor.method = 'precomputed'
        precomputed_output, precomputed_time = processor.process_events(events_np)
        precomputed_mse = mse(original_output, precomputed_output)
        precomputed_speedup = original_time / precomputed_time

        processor.method = 'scipy_gaussian'
        scipy_output, scipy_time = processor.process_events(events_np)
        scipy_mse = mse(original_output, scipy_output)
        scipy_speedup = original_time / scipy_time

        processor.method = 'dense_torch'
        torch_output, torch_time = processor.process_events(events_np)
        torch_mse = mse(original_output, torch_output)
        torch_speedup = original_time / torch_time

        processor.method = 'separable_torch'
        sep_output, sep_time = processor.process_events(events_np)
        sep_mse = mse(original_output, sep_output)
        sep_speedup = original_time / sep_time

        print(f"Num Points: {num_points}")
        print(f"Original Time: {original_time:.4f}s")
        print(f"Precomputed Time: {precomputed_time:.4f}s, MSE: {precomputed_mse:.6f}, Speedup: {precomputed_speedup:.4f}")
        print(f"Scipy Gaussian Time: {scipy_time:.4f}s, MSE: {scipy_mse:.6f}, Speedup: {scipy_speedup:.4f}")
        print(f"Dense Torch Time: {torch_time:.4f}s, MSE: {torch_mse:.6f}, Speedup: {torch_speedup:.4f}")
        print(f"Separable Torch Time: {sep_time:.4f}s, MSE: {sep_mse:.6f}, Speedup: {sep_speedup:.4f}")