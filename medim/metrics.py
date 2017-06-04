import numpy as np


def dice_score(y_pred, target):
    """Dice score between two binary masks."""
    return 2 * np.sum(y_pred * target) / (np.sum(y_pred) + np.sum(target))
