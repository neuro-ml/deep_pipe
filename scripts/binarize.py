import os

import numpy as np

from dpipe.config import get_config, ResourceManager
from utils import load_by_id

if __name__ == '__main__':
    config = get_config('ids_path', 'predictions_path', 'thresholds_path',
                        'binary_predictions_path')
    resource_manager = ResourceManager(config)

    binary_predictions_path = resource_manager['binary_predictions_path']
    ids = resource_manager['ids']
    thresholds_path = config['thresholds_path']
    predict_path = config['predictions_path']

    thresholds = np.load(thresholds_path)
    channels = len(thresholds)
    os.makedirs(binary_predictions_path)

    for identifier in ids:
        y = load_by_id(predict_path, identifier)
        assert len(y) == channels

        for i in range(channels):
            y[i] = y[i] > thresholds[i]

        np.save(os.path.join(binary_predictions_path, str(identifier)), y)
        del y
