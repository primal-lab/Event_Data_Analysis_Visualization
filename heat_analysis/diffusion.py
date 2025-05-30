"""
Diffusion processing utilities.
Contains functions for applying diffusion and generating diffusion kernels.
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')  # Use non-interactive backend
import imageio.v3 as iio


def heat_kernel(x, y, t, k):
    denom = 4 * np.pi * k * (t + 1e-8)
    exponent = -(x**2 + y**2) / (4 * k * t + 1e-8)
    H =  (1 / denom) * np.exp(exponent)
    return H/H.sum()


def heat_kernel_gradient(x, y, t, k):
    eps = 1e-8
    H = heat_kernel(x, y, t, k)

    dH_dx = H * (-x / (2 * k * (t + eps)))
    dH_dy = H * (-y / (2 * k * (t + eps)))
    # print(f"x: {x}, y: {y} t: {t} H_max : {H[x, y]} Alpha: {k:.2e} Extra Grad: {-x/ (2 * k * (t + eps))} Max dH_dx: {dH_dx[x, y]}")
    return dH_dx, dH_dy

def generate_heat_kernel_3d_np(T, H, W, k=0.05):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    kernel = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        kernel[t - 1] = heat_kernel(X, Y, t/T, k)
        # if t%300==0: print(f"At time t: {t/T:.2e} Kenel Sum: {np.sum(kernel[t - 1])} and max: {kernel[t - 1].max()}")
    return kernel

def generate_heat_kernel_gradient_3d_np(T, H, W, k=0.05):
    
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    dH_dx_3d = np.zeros((T, H, W), dtype=np.float32)
    dH_dy_3d = np.zeros((T, H, W), dtype=np.float32)
    for t in range(1, T + 1):
        dH_dx, dH_dy = heat_kernel_gradient(X, Y, t/T, k)
        dH_dx_3d[t - 1] = dH_dx.astype(np.float32)
        dH_dy_3d[t - 1] = dH_dy.astype(np.float32)
    
    return dH_dx_3d, dH_dy_3d
