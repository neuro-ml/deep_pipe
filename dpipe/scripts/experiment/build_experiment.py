import os
import json

from dpipe.config import parse_config, get_parser
from dpipe.config import config_dataset, config_split, config_build_experiment

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('-ep', '--experiments_path')
    parser.add_argument('-sp', '--scripts_path')
    config = parse_config(parser)

    experiments_path = os.path.abspath(config['experiments_path'])
    scripts_path = config['scripts_path']

    dataset = config_dataset(config)
    split = config_split(config, dataset)
    build_experiment = config_build_experiment(config)

    build_experiment(split, experiments_path)

    with open(os.path.join(experiments_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    with open(os.path.join(experiments_path, 'paths.json'), 'w') as f:
        json.dump({'scripts_path': scripts_path},
                  f, indent=2, sort_keys=True)
