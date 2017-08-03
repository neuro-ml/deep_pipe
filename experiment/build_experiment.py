import os

from dpipe.config import parse_config, get_parser
from dpipe.config import config_dataset, config_split, config_experiment


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('-ep', '--experiment_path')
    config = parse_config(parser)

    dataset = config_dataset(config)
    split = config_split(config, dataset=dataset)

    config_path = config['config_path']
    experiment_path = os.path.realpath(config['experiment_path'])

    config_experiment(config, experiment_path=experiment_path, split=split,
                      config_path=config_path)
