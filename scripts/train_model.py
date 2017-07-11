import numpy as np

from experiments.config import config_dataset, config_optimizer, \
    config_batch_iter, config_model, config_train
from experiments.default_parser import parse_config, get_default_parser
from experiments.dl import ModelController

if __name__ == '__main__':
    # parser
    parser = get_default_parser()
    parser.add_argument('-tid', '--train_ids_path')
    parser.add_argument('-vid', '--val_ids_path')
    parser.add_argument('-lp', '--log_path')
    parser.add_argument('-smp', '--save_model_path')
    config = parse_config(parser)

    # find paths
    train_ids_path = config['train_ids_path']
    val_ids_path = config['val_ids_path']
    log_path = config['log_path']
    save_model_path = config['save_model_path']

    train_ids = np.loadtxt(train_ids_path, str, delimiter='\n')
    val_ids = np.loadtxt(val_ids_path, str, delimiter='\n')

    # building objects
    dataset = config_dataset(config)
    train_batch_iter = config_batch_iter(config, ids=train_ids, dataset=dataset)
    optimizer = config_optimizer(config)
    model = config_model(config, optimizer=optimizer,
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)
    train = config_train(config)

    with ModelController(model, log_path) as mc:
        train(mc, train_batch_iter, val_ids, dataset)
        model.save(save_model_path)
