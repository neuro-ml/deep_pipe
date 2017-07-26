import os

import numpy as np

from dpipe.config import config_dataset
from dpipe.config.default_parser import get_config
from dpipe.medim.metrics import dice_score as dice
from utils import read_lines, load_by_id

if __name__ == '__main__':
    config = get_config('metrics_path', 'ids_path', 'thresholds_path',
                        'predictions_path')

    metrics_path = config['metrics_path']
    ids_path = config['ids_path']
    thresholds_path = config['thresholds_path']
    predictions_path = config['predictions_path']
    dataset = config_dataset(config)

    ids = read_lines(ids_path)
    thresholds = np.load(thresholds_path)

    channels = dataset.n_chans_msegm

    dices = []
    for identifier in ids:
        y_true = dataset.load_msegm(identifier)
        y = load_by_id(predictions_path, identifier)

        dices.append([dice(y[i] > thresholds[i], y_true[i])
                      for i in range(channels)])

        # saving some memory
        del y, y_true

    np.save(metrics_path, dices)
