import numpy as np

from dpipe.config import get_config, get_resource_manager
from dpipe.medim.metrics import dice_score as dice
from utils import load_by_id

if __name__ == '__main__':
    resource_manager = get_resource_manager(get_config(
        'config_path', 'ids_path', 'thresholds_path', 'predictions_path'
    ))

    threshold_path = resource_manager['thresholds_path']
    predictions_path = resource_manager['predictions_path']
    ids = resource_manager['ids']
    dataset = resource_manager['dataset']

    channels = dataset.n_chans_msegm
    dices = [[] for _ in range(channels)]
    thresholds = np.linspace(0, 1, 20)
    for identifier in ids:
        y_true = dataset.load_msegm(identifier)
        y = load_by_id(predictions_path, identifier)

        # get dice with individual threshold for each channel
        for i in range(channels):
            dices[i].append([dice(y[i] > thr, y_true[i])
                             for thr in thresholds])

        # saving some memory
        del y, y_true
    dices = np.asarray(dices)
    identifier = dices.mean(axis=1).argmax(axis=1)
    final = thresholds[identifier]

    np.save(threshold_path, final)
