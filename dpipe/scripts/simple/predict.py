import os

import numpy as np

from dpipe.config import config_dataset, config_model
from dpipe.config.default_parser import get_config
from dpipe.modules.dl import ModelController, Optimizer
from utils import read_lines

if __name__ == '__main__':
    config = get_config('ids_path', 'restore_model_path',
                        'predictions_path', 'log_path')

    results_path = config['predictions_path']
    ids_path = config['ids_path']
    restore_model_path = config['restore_model_path']
    log_path = config['log_path']
    dataset = config_dataset(config)
    model = config_model(config, optimizer=Optimizer(),
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)

    ids = read_lines(ids_path)
    os.makedirs(results_path)

    with ModelController(model, log_path, restore_model_path) as mc:
        for identifier in ids:
            x = dataset.load_mscan(identifier)
            y = mc.predict_object(x)

            np.save(os.path.join(results_path, str(identifier)), y)

            # saving some memory
            del x, y
