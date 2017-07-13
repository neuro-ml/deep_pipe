import argparse
import json
import pprint
import re

from dpipe.config.config import default_config
from dpipe.modules.datasets.config import dataset_name2default_params

__all__ = ['parse_config', 'get_parser', 'get_config']

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
        if value is not None:
            config[arg] = value
    _parse_unknown(unknown, config)

    # default values
    _merge_configs(config, default_config)
    # module-specific:
    for module, params in module_type2default_params_mapping.items():
        field_name = f'{module}__params'
        if config.get(module, None) is not None:
            # raise ValueError(f'"{module}" parameter not specified')
            _merge_configs(config[field_name], params[config[module]])

    # final checks
    # for arg, value in args._get_kwargs():
    #     if value is None and config.get(arg) is None and arg != 'config_path':
    #         raise ValueError(f'"{arg}" parameter not specified')

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


def get_parser(*additional_params) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-cp', '--config', dest='config_path')

    for param in additional_params:
        kwargs = available_params[param]
        if type(kwargs) is dict:
            args = kwargs.pop('names')
        else:
            args, kwargs = kwargs, {}
        kwargs['dest'] = param
        if '--' + param not in args:
            args = args + ['--' + param]
        parser.add_argument(*args, **kwargs)
    return parser


def get_config(*additional_params) -> dict:
    return parse_config(get_parser(*additional_params))


available_params = {
    'batch_iter': ['-bi', '--iter'],
    'batch_size': dict(names=['-bs', '--batch_size'], type=int),

    'dataset': ['-ds', '--dataset'],
    'dataset_cached': dict(names=['--chached'], action='store_true',
                           default=False,
                           help='whether the dataset is chached'),

    'model': ['-m', '--model'],
    'model_path': ['-mp', '--model_path'],
    'save_model_path': ['-smp', '--save_model_path'],
    'predictions_path': ['-pp', '--predictions_path'],

    'train_ids_path': ['-tid', '--train_ids_path'],
    'val_ids_path': ['-vid', '--val_ids_path'],
    'ids_path': ['-ip', '--ids_path'],

    'log_dir': ['-ld', '--log_dir'],
    'thresholds_path': ['-thp', '--thresholds_path'],
    'results_path': ['-p'],
}

if __name__ == '__main__':
    pprint.pprint(parse_config(get_parser()))
