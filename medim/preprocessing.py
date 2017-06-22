import numpy as np


def normalize_scan(scan, mean=True, std=True, drop_percentile: int=None):
    """Normalize scan to make mean and std equal to (0, 1) if stated.
    Supports robust estimation with drop_percentile."""
    if drop_percentile is not None:
        bottom = np.percentile(scan, drop_percentile)
        top = np.percentile(scan, 100 - drop_percentile)
        
        mask = (scan > bottom) & (scan < top)
        vals = scan[mask]
    else:
        vals = scan.flatten()

    if mean:
        scan = scan - vals.mean()

    if std:
        scan = scan / vals.std()

    return np.array(scan, dtype=np.float32)


def normalize_mscan(mscan, mean=True, std=True, drop_percentile: int=None):
    """Normalize mscan to make mean and std equal to (0, 1) if stated.
    Supports robust estimation with drop_percentile."""
    new_mscan = np.zeros_like(mscan, dtype=np.float32)
    for i in range(len(mscan)):
        new_mscan[i] = normalize_scan(mscan[i], mean=mean, std=std,
                                      drop_percentile=drop_percentile)
    return new_mscan
