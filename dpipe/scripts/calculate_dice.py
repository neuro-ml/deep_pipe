import numpy as np

from dpipe.config import parse_config, get_default_parser
from dpipe.config import config_dataset, config_model, config_optimizer
from dpipe.modules.dl import ModelController
from dpipe.medim.metrics import dice_score as dice
from utils import read_lines

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('-dp', '--dices_path')
    parser.add_argument('-ip', '--ids_path')
    parser.add_argument('-mp', '--model_path')
    parser.add_argument('-thp', '--thresholds_path')
    config = parse_config(parser)

    dices_path = config['dices_path']
    ids_path = config['ids_path']
    model_path = config['model_path']
    thresholds_path = config['thresholds_path']

    ids = read_lines(ids_path)
    thresholds = np.load(thresholds_path)

    dataset = config_dataset(config)
    optimizer = config_optimizer(config)
    model = config_model(config, optimizer=optimizer,
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)

    n_chans_msegm = dataset.n_chans_msegm

    dices = []
    with ModelController(model, dices_path, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y_true = dataset.load_msegm(id)
            y = mc.predict_object(x)

            dices.append([dice(y[i] > thresholds[i], y_true[i])
                          for i in range(n_chans_msegm)])

            # saving some memory
            del x, y, y_true

    np.save(dices_path, dices)
