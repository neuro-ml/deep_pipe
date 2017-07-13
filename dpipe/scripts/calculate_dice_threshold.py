import os

import numpy as np

from dpipe.config import config_dataset
from dpipe.config.default_parser import get_config
from dpipe.medim.metrics import dice_score as dice
from utils import read_lines

if __name__ == '__main__':
    config = get_config('ids_path', 'thresholds_path',
                        'predictions_path', 'dataset')

    threshold_path = config['thresholds_path']
    predictions_path = config['predictions_path']
    ids_path = config['ids_path']
    dataset = config_dataset(config)

    ids = read_lines(ids_path)
    channels = dataset.n_chans_msegm
    dices = [[] for _ in range(channels)]
    thresholds = np.linspace(0, 1, 20)
    for id in ids:
        y_true = dataset.load_msegm(id)
        y = np.load(os.path.join(predictions_path, str(id)))

        # get dice with individual threshold for each channel
        for i in range(channels):
            dices[i].append([dice(y[i] > thr, y_true[i])
                             for thr in thresholds])

        # saving some memory
        del y, y_true
    dices = np.asarray(dices)
    idx = dices.mean(axis=1).argmax(axis=1)
    final = thresholds[idx]

    np.save(threshold_path, final)
