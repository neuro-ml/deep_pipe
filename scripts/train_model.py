from utils import read_lines

from dpipe.config import config_dataset, config_batch_iter_factory, \
    config_data_loader
from dpipe.config.config_tf import config_model, config_train, \
    config_model_controller
from dpipe.config.default_parser import get_config

from dpipe.dl import ModelController

if __name__ == '__main__':
    config = get_config('train_ids_path', 'val_ids_path', 'log_path',
                        'save_model_path', 'restore_model_path', 'save_on_quit')

    train_ids_path = config['train_ids_path']
    val_ids_path = config['val_ids_path']
    log_path = config['log_path']
    save_model_path = config['save_model_path']
    restore_model_path = config.get('restore_model_path', None)
    save_on_quit = config.get('save_on_quit')

    train_ids = read_lines(train_ids_path)
    val_ids = read_lines(val_ids_path)

    dataset = config_dataset(config)
    data_loader = config_data_loader(config, dataset)
    train_batch_iter = config_batch_iter_factory(config, ids=train_ids,
                                                 data_loader=data_loader)
    model = config_model(config, data_loader)
    model_controller = ModelController(model, log_path, restore_model_path)
    train = config_train(config, train_batch_iter, data_loader,
                         model_controller, val_ids)

    with model_controller:
        try:
            train()
            model.save(save_model_path)
        except KeyboardInterrupt:
            if save_on_quit:
                model.save(save_model_path)
            else:
                raise
