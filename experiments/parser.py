import os
import copy
import json
import pprint
import argparse

from experiments.config import default_config
from experiments.datasets.config import dataset_name2default_params

__all__ = ['parse_config']

# TODO Add info about arguments vs config settings differences after discussion
description = """
    Config system.
    First we read default config file, defined in the
    library root, then parse config file, if it was presented, then we
    parse command arguments and finally, we fill parameters specific for each
    particular object (for instance, dataset path) with default ones,
    if they were not provided before.
    Command arguments are not supported yet."""


module_type2default_params_mapping = {
    'dataset': dataset_name2default_params
}


def parse_config():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config_file', nargs='?',
                        help='json file with config')
    args = parser.parse_args()
    config_file = args.config_file

    # We start with default config
    config = copy.deepcopy(default_config)

    # Update it with file config
    if config_file is not None:
        with open(config_file, 'r') as f:
            new_config = json.load(f)
        config = combine_configs(config, new_config)

    # Update object params with default ones for this specific object
    for module_type, default_params_mapping in \
            module_type2default_params_mapping.items():
        field_name = f'{module_type}__params'
        try:
            config[field_name] = get_updated_params(
                default_params_mapping[config[module_type]], config[field_name])
        except KeyError as error:
            print()
            raise ValueError(f"Error, while trying to parse {module_type}") \
                from error

    results_path = config['results_path']
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return config


def combine_configs(base_config: dict, new_config: dict):
    config = copy.deepcopy(base_config)
    config.update(new_config)

    inner_configs = filter(lambda x: x.endswith('__config'),
                           set(base_config).intersection(set(new_config)))
    for key in inner_configs:
        assert type(base_config[key]) is dict and type(new_config[key]) is dict
        value = copy.deepcopy(base_config[key])
        value.update(new_config[key])
        config[key] = value

    return config


def get_updated_params(default_params, new_params):
    params = copy.deepcopy(default_params)
    params.update(new_params)
    return params


if __name__ == '__main__':
    config = parse_config()
    pprint.pprint(config)
