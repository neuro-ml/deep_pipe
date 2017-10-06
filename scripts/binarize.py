import os

import numpy as np

from dpipe.config import get_args, get_resource_manager
from utils import load_by_id

if __name__ == '__main__':
    rm = get_resource_manager(**get_args(
        'config_path', 'ids_path', 'predictions_path', 'thresholds_path',
        'binary_predictions_path'
    ))

    thresholds = np.load(rm.thresholds_path)
    channels = len(thresholds)
    os.makedirs(rm.binary_predictions_path)

    for identifier in rm.ids:
        y = load_by_id(rm.predictions_path, identifier)
        assert len(y) == channels

        y = y > thresholds[:, None, None, None]

        np.save(os.path.join(rm.binary_predictions_path, str(identifier)), y)
        del y
