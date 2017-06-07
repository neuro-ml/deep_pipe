import numpy as np


def dice_score(y_pred, target):
    """Dice score between two binary masks."""
    assert y_pred.dtype == target.dtype == np.bool

    num = 2 * np.sum(y_pred * target)
    den = np.sum(y_pred) + np.sum(target)

    return 1 if den == 0 else num / den
