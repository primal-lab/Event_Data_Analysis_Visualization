"""
Visualization module.
Contains functions for creating various visualizations of event data.
"""

from .video import make_side_by_side_video
from .heatmap import generate_heatmap, heat_kernel, generate_heat_kernel_3d_np

__all__ = [
    'make_side_by_side_video',
    'create_heatmap',
    'create_quiver_plot'
] 