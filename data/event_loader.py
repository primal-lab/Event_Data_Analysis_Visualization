"""
Event data loading and parsing utilities.
Contains functions for loading event data from files and parsing metadata.
"""

import numpy as np


def parse_meta(meta_path):
    with open(meta_path, 'r') as f:
        data = eval(f.read())
    frames = data.get("frames", [])
    return sorted([(f['ts'], f['classical_frame']) for f in frames])

def load_events_fast(event_path):
    data = np.loadtxt(event_path, dtype=np.float32)
    data = data[np.isin(data[:, 3], [-1, 0, 1])]
    data[:, 1:4] = np.round(data[:, 1:4])
    return data