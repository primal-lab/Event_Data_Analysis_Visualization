"""
Data loading and processing module.
Contains utilities for loading event data and RGB frames.
"""

from .event_loader import load_events_fast, parse_meta
from .frame_loader import process_frame

__all__ = [
    'load_events_fast',
    'parse_meta',
    'process_frames'
] 