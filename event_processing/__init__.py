"""
Event processing module.
Contains functions for event tensor generation and dataset implementation.
"""

from .event_tensor import build_event_tensor
from .event_dataset import EventDataset

__all__ = [
    'build_event_tensor',
    'EventDataset'
]