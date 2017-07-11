import numpy as np

from dpipe.config import parse_config, get_default_parser
from dpipe.config import config_dataset, config_model
from dpipe.modules.dl import ModelController, Optimizer
from dpipe.medim.metrics import dice_score as dice

# not sure if I need main
if __name__ == '__main__':
    # parser
    parser = get_default_parser()
    parser.add_argument('-rp', '--results', dest='results_path')
    parser.add_argument('-i', '--ids_path')
    parser.add_argument('-mp', '--model_path')
    parser.add_argument('-th', '--thresholds_path')
    config = parse_config(parser)

    # building objects
    results_path = config['results_path']
    thresholds_path = config['thresholds_path']
    dataset = config_dataset(config)
    model = config_model(config, optimizer=Optimizer(),
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)
    model_path = config['model_path']
    ids = config['ids_path']
    ids = np.loadtxt(ids, str, delimiter='\n')
    thresholds = np.load(thresholds_path)
    channels = dataset.n_chans_msegm

    dices = []
    with ModelController(model, results_path, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y_true = dataset.load_msegm(id)
            y = mc.predict_object(x)

            dices.append([dice(y[i] > thresholds[i], y_true[i])
                          for i in range(channels)])

            # saving some memory
            del x, y, y_true

    np.save(results_path, dices)
