from itertools import product

import numpy as np


def divide(mscan, padding, n_parts):
    """Divides padded mscan (should be padded beforehand)
    into multiple parts of about the same shape according to the n_parts array
    and padding."""
    padding = np.array(padding)
    n_parts = np.array(n_parts)
    steps = (np.array(mscan.shape[1:]) - 2 * padding) // n_parts
    steps += ((np.array(mscan.shape[1:]) - 2 * padding) % n_parts) > 0

    parts = []
    for idx in product(range(n_parts[0]), range(n_parts[1]), range(n_parts[2])):
        lb = np.array(idx) * steps
        slices = list(map(slice, lb, lb + steps + 2 * padding))
        parts.append(mscan[[...] + slices])

    return parts


def combine(parts, n_parts):
    """Combines parts of big mscan into one big mscan, according to n_parts"""
    parts = list(parts)
    n_parts = np.array(n_parts)

    xs = []
    for i in range(n_parts[0]):
        ys = []
        for j in range(n_parts[1]):
            zs = []
            for k in range(n_parts[2]):
                zs.append(parts.pop(0))
            y = np.concatenate(zs, axis=3)
            ys.append(y)
        x = np.concatenate(ys, axis=2)
        xs.append(x)
    return np.concatenate(xs, axis=1)