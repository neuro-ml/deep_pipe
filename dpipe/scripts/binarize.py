import os

import numpy as np

from dpipe.config.default_parser import get_config
from utils import read_lines, load_by_id

if __name__ == '__main__':
    config = get_config('ids_path', 'predictions_path', 'thresholds_path',
                        'results_path')

    results_path = config['results_path']
    ids_path = config['ids_path']
    thresholds_path = config['thresholds_path']
    predict_path = config['predictions_path']

    ids = read_lines(ids_path)
    thresholds = np.load(thresholds_path)
    channels = len(thresholds)
    os.makedirs(results_path)

    for id in ids:
        y = load_by_id(predict_path, id)
        assert len(y) == channels

        for i in range(channels):
            y[i] = y[i] > thresholds[i]

        np.save(os.path.join(results_path, str(id)), y)
        del y
