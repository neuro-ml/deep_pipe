import argparse
import json
import os
import pprint
import re

from dpipe.config.config import default_config
from dpipe.modules.datasets.config import dataset_name2default_params

__all__ = ['parse_config', 'get_default_parser']

# TODO Add info about arguments vs config settings differences after discussion
description = """
    Config system.
    First we read default config file, defined in the
    library root, then parse config file, if it was presented, then we
    parse command arguments and finally, we fill parameters specific for each
    particular object (for instance, dataset path) with default ones,
    if they were not provided before."""

module_type2default_params_mapping = {
    'dataset': dataset_name2default_params
}


def parse_config(parser: argparse.ArgumentParser) -> dict:
    args, unknown = parser.parse_known_args()

    config = {}
    # config file:
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)

    # batch_size is the only exception
    if 'batch_size' in args and args.batch_size is not None:
        config['batch_iter__params']['batch_size'] = args.batch_size
        args.batch_size = None

    # console arguments:
    for arg, value in args._get_kwargs():
        config[arg] = value
    _parse_unknown(unknown, config)

    # default values
    _merge_configs(config, default_config)
    # module-specific:
    for module, params in module_type2default_params_mapping.items():
        field_name = f'{module}__params'
        if config[module] is None:
            raise ValueError(f'"{module}" parameter not specified')
        _merge_configs(config[field_name], params[config[module]])

    # final checks
    for arg, value in args._get_kwargs():
        if value is None and config.get(arg) is None and arg != 'config_path':
            raise ValueError(f'"{arg}" parameter not specified')

    results_path = config['results_path']
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return config


def _parse_unknown(unknown, config):
    i = 0
    argument = re.compile('^--((\w+__)+\w+)$')
    while i < len(unknown):
        arg = argument.match(unknown[i])
        if not arg:
            raise ValueError(f'Unknown argument "{unknown[i]}"')
        i += 1
        arg = arg.group(1)
        # reaching the top level
        names = arg.split('__')
        names[0] += '__params'
        temp = config
        for name in names[:-1]:
            temp = config.setdefault(name, {})
        # parsing the value: for now it can be str, int or bool
        if i >= len(unknown) or unknown[i][:2] == '--':
            # if the value is omitted, then its value is True
            value = True
        else:
            value = unknown[i]
            try:
                value = int(value)
            except ValueError:
                pass
            if value in ['True', 'False']:
                value = value == 'True'
            i += 1

        temp[names[-1]] = value


def _merge_configs(destination: dict, source: dict):
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
        else:
            if type(value) is dict:
                _merge_configs(destination[key], value)


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-cf', '--config', dest='config_path')
    return parser


# default_parser.add_argument('-bi', '--iter', dest='batch_iter',
#                             help='the batch iterator')
# default_parser.add_argument('-bs', '--batch_size', type=int,
#                             help='the batch size')
# default_parser.add_argument('-ds', '--dataset')
# default_parser.add_argument('--chached', action='store_true',
#                             dest='dataset_cached', default=False,
#                             help='whether the dataset is chached')
# default_parser.add_argument('-m', '--model')
# default_parser.add_argument('-p', '--path', dest='results_path',
#                             help='results path')
# default_parser.add_argument('-s', '--splitter')

if __name__ == '__main__':
    pprint.pprint(parse_config(get_default_parser()))
