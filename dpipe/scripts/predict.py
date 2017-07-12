import os

import numpy as np

from dpipe.config import parse_config, get_default_parser
from dpipe.config import config_dataset, config_model
from dpipe.modules.dl import ModelController, Optimizer
from utils import read_lines

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('-pp', '--predictions_path')
    parser.add_argument('-ip', '--ids_path')
    parser.add_argument('-mp', '--model_path')
    config = parse_config(parser)

    results_path = config['predictions_path']
    ids_path = config['ids_path']
    model_path = config['model_path']

    ids = read_lines(ids_path)

    dataset = config_dataset(config)
    model = config_model(config, optimizer=Optimizer(),
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)

    with ModelController(model, results_path, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y = mc.predict_object(x)
            np.save(os.path.join(results_path, str(id)), y)
            # saving some memory
            del x, y
