import re
import copy
import os
import json
import pprint
import argparse

from dpipe.config.config import default_config
from dpipe.modules.datasets.config import dataset_name2default_params

__all__ = ['parse_config', 'get_parser', 'get_config']

# you can pas either a dict with params, or a just an array with names
# the param's name is added to the names array, if it is not present there

# TODO Add info about arguments vs config settings differences after discussion
description = """
    Config system.
    First we read default config file, defined in the
    library root, then parse config file, if it was presented, then we
    parse command arguments and finally, we fill parameters specific for each
    particular object (for instance, shadowed path) with default ones,
    if they were not provided before."""

module_type2default_params_mapping = {
    'dataset': dataset_name2default_params
}


def head(xs):
    return xs[0]


def get_short_name(name: str):
    return ''.join(map(head, name.split('_')))


# Parameter aname1_bname2_..._cnameN for simple scrips can be provided as
# --aname1_bname2_..._cnameN or -ab...c
simple_script_params = {
    name: (f'-{get_short_name(name)}', f'--{name}')
    for name in (
    'train_ids_path', 'val_ids_path', 'ids_path',
    'save_model_path', 'restore_model_path',
    'predictions_path', 'binary_predictions_path',
    'thresholds_path', 'metrics_path',
    'log_path',
)
}

config_params = {
    'batch_iter': ['-bi', '--iter'],
    'batch_size': dict(names=['-bs', '--batch_size'], type=int),

    'dataset': ['-ds', '--dataset'],
    'dataset_cached': dict(names=['--chached'], action='store_true',
                           default=False,
                           help='whether the shadowed is chached'),

    'model': ['-m'],
}

available_params = copy.deepcopy(config_params)
available_params.update(simple_script_params)


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
    for module, default_params in module_type2default_params_mapping.items():
        field_name = f'{module}__params'
        params = config.get(field_name, {})
        _merge_configs(params, default_params[config[module]])
        config[field_name] = params

    # TODO: for now a dirty hack:
    for name in ('save_model_path', 'restore_model_path'):
        if config.get(name, None) is not None:
            config[name] = os.path.join(config[name], 'dump')

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


if __name__ == '__main__':
    pprint.pprint(parse_config(get_parser()))
