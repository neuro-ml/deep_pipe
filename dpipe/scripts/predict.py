import os

import numpy as np

from dpipe.config import config_dataset, config_model
from dpipe.config.default_parser import get_config
from dpipe.modules.dl import ModelController, Optimizer
from utils import read_lines

if __name__ == '__main__':
    config = get_config('ids_path', 'thresholds_path', 'model_path', 'model'
                        'predictions_path', 'dataset', 'log_dir')

    results_path = config['predictions_path']
    ids_path = config['ids_path']
    model_path = config['model_path']
    log_dir = config['log_dir']
    dataset = config_dataset(config)
    model = config_model(config, optimizer=Optimizer(),
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)

    ids = read_lines(ids_path)

    with ModelController(model, log_dir, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y = mc.predict_object(x)

            np.save(os.path.join(results_path, str(id)), y)

            # saving some memory
            del x, y
