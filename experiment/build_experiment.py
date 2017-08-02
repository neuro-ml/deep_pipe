import os
import json
import shutil

import numpy as np

from dpipe.config import parse_config, get_parser, get_paths
from dpipe.config import config_dataset, config_split


def build_flat_structure(split, experiments_path: str, makefile_path: str):
    for i, ids in enumerate(split):
        local = os.path.join(experiments_path, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(makefile_path, os.path.join(local, 'Snakefile'))

        for val, file in zip(ids, ('train', 'val', 'test')):
            file = os.path.join(local, f'{file}_ids')
            np.savetxt(file, val, delimiter='\n', fmt='%s')


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('-ep', '--experiments_path')
    config = parse_config(parser)

    dataset = config_dataset(config)
    split = config_split(config, dataset)

    experiments_path = os.path.realpath(config['experiments_path'])

    paths = get_paths()
    makefile_path = os.path.join(paths['makefiles'], config['makefile'])
    scripts_path = paths['scripts']

    build_flat_structure(split, experiments_path, makefile_path)

    with open(os.path.join(experiments_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    with open(os.path.join(experiments_path, 'paths.json'), 'w') as f:
        json.dump({'scripts': scripts_path}, f, indent=2, sort_keys=True)
