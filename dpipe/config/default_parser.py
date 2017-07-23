import copy
import os
import json
import pprint
import argparse
from collections import ChainMap


from dpipe.modules.datasets.config import dataset_name2default_params

__all__ = ['parse_config', 'get_parser', 'get_config']

# you can pas either a dict with params, or a just an array with names
# the param's name is added to the names array, if it is not present there

description = "Description"

module_type2default_params_mapping = {
    'dataset': dataset_name2default_params
}


def get_short_name(name: str) -> str:
    """long_informative_name -> lip"""
    return ''.join(map(lambda x: x[0], name.split('_')))


# Simple parameters, similar across scripts, have to be provided, if mentioned
common_script_params = {
    name: (f'-{get_short_name(name)}', f'--{name}')
    for name in (
        'train_ids_path', 'val_ids_path', 'ids_path',
        'save_model_path', 'restore_model_path',
        'predictions_path', 'binary_predictions_path',
        'thresholds_path', 'metrics_path',
        'log_path',
    )
}

# Complex parameters
# Defaults are allowed only for flags
additional_params = {
    'batch_size': dict(names=['-bs', '--batch_size'], type=int),
    'save_on_quit': dict(
        names=['--save'], action='store_true',
        help='whether to save the model after ctrl+c is pressed'),
}

# Empty dict to protect all the rest from writing
available_params = ChainMap({}, common_script_params, additional_params)


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
