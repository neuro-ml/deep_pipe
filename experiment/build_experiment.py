import os
import json
import shutil

import numpy as np

from dpipe.config import parse_config, get_parser
from dpipe.config import config_dataset, config_split


def build_flat_structure(split, experiment_dir: str, makefile_path: str):
    for i, ids in enumerate(split):
        local = os.path.join(experiment_dir, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(makefile_path, os.path.join(local, 'Snakefile'))

        for val, file in zip(ids, ('train', 'val', 'test')):
            file = os.path.join(local, f'{file}_ids')
            np.savetxt(file, val, delimiter='\n', fmt='%s')


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

    with open(os.path.join(experiments_path, 'iitp_paths.json'), 'w') as f:
        json.dump({'scripts_path': scripts_path},
                  f, indent=2, sort_keys=True)
