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

    channels = dataset.n_chans_msegm
    dices = [[] for _ in range(channels)]
    thresholds = np.linspace(0, 1, 20)
    with ModelController(model, results_path, model_path) as mc:
        for id in ids:
            x = dataset.load_mscan(id)
            y_true = dataset.load_msegm(id)
            y = mc.predict_object(x)

            # get dice with individual threshold for each channel
            for i in range(channels):
                dices[i].append([dice(y[i] > thr, y_true[i])
                                 for thr in thresholds])

            # saving some memory
            del x, y, y_true
    dices = np.asarray(dices)
    idx = dices.mean(axis=1).argmax(axis=1)
    final = thresholds[idx]

    np.save(results_path, final)
