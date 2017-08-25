import numpy as np

from dpipe.config import get_config, get_resource_manager
from dpipe.medim.metrics import dice_score as dice
from utils import load_by_id

if __name__ == '__main__':
    resource_manager = get_resource_manager(get_config(
        'config_path', 'metrics_path', 'ids_path', 'thresholds_path',
        'predictions_path'
    ))

    metrics_path = resource_manager['metrics_path']
    ids = resource_manager['ids']
    thresholds_path = resource_manager['thresholds_path']
    predictions_path = resource_manager['predictions_path']
    dataset = resource_manager['dataset']

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
