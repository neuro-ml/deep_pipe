import os

import numpy as np

from experiments.config import config_dataset, config_model
from experiments.default_parser import parse_config, get_default_parser
from experiments.dl import ModelController, Optimizer

# not sure if I need main
if __name__ == '__main__':
    # parser
    parser = get_default_parser()
    parser.add_argument('-rp', '--results', dest='results_path')
    parser.add_argument('-i', '--ids_path')
    parser.add_argument('-mp', '--model_path')
    config = parse_config(parser)

    # building objects
    # TODO: probably we need an object builder
    results_path = config['results_path']
    dataset = config_dataset(config)
    model = config_model(config, optimizer=Optimizer(),
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)
    model_path = config['model_path']
    ids = config['ids_path']
    ids = np.loadtxt(ids, str, delimiter='\n')

    with ModelController(model, results_path, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y = mc.predict_object(x)
            np.save(os.path.join(results_path, str(id)), y)
            # saving some memory
            del x, y
