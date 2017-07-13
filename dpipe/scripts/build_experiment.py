import os
import json

from dpipe.config import parse_config, get_parser
from dpipe.config import config_dataset, config_split, config_build_experiment

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('-ed', '--experiment_dir')
    config = parse_config(parser)

    experiment_dir = config['experiment_dir']

    dataset = config_dataset(config)
    split = config_split(config, dataset)
    build_experiment = config_build_experiment(config)

    build_experiment(split, experiment_dir)
    #os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
