import numpy as np


def normalize_scan(scan, drop_percentile=2):
    """Normalize scan to make mean and std equal to (0, 1).
    We don't use 0-s to esimate stats."""
    if drop_percentile:
        bottom = np.percentile(scan, drop_percentile)
        top = np.percentile(scan, 100 - drop_percentile)
        
        mask = (scan > bottom) & (scan < top)
        vals = scan[mask]
    else:
        vals = scan.flatten()
    
    mean = vals.mean()
    std = vals.std()
    
    scan = (scan - mean) / std
    return np.array(scan, dtype=np.float32)


def normalize_mscan(mscan, drop_percentile=2):
    """Normalize mscan to make mean and std equal to (0, 1).
    We don't use 0-s to esimate stats."""
    new_mscan = np.zeros(mscan.shape, dtype=np.float32)
    for i in range(len(mscan)):
        new_mscan[i] = normalize_scan(mscan[i], drop_percentile)
    return new_mscan
