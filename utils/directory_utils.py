"""
Utility functions for directory operations.
"""
from pathlib import Path

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    dirs = ['NPY', 'Plots', 'videos']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True) 