from dpipe.config import config_dataset, config_batch_iter, config_model, \
    config_train
from dpipe.config.default_parser import get_config
from dpipe.modules.dl import ModelController

from utils import read_lines

if __name__ == '__main__':
    config = get_config('train_ids_path', 'val_ids_path', 'log_path',
                        'save_model_path', 'restore_model_path')

    train_ids_path = config['train_ids_path']
    val_ids_path = config['val_ids_path']
    log_path = config['log_path']
    save_model_path = config['save_model_path']
    restore_model_path = config.get('restore_model_path', None)

    train_ids = read_lines(train_ids_path)
    val_ids = read_lines(val_ids_path)

    dataset = config_dataset(config)
    train_batch_iter = config_batch_iter(config, ids=train_ids, dataset=dataset)
    model = config_model(config, dataset)
    train = config_train(config)

    with ModelController(model, log_path, restore_model_path) as mc:
        train(mc, train_batch_iter, val_ids, dataset)
        model.save(save_model_path)
