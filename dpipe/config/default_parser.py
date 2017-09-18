import json
import argparse

__all__ = ['parse_config', 'get_parser', 'get_config']


def get_short_name(name: str) -> str:
    """long_informative_name -> lin"""
    return ''.join(map(lambda x: x[0], name.split('_')))


def parse_config(parser: argparse.ArgumentParser) -> dict:
    args, unknown = parser.parse_known_args()

    # Load config file
    try:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    except AttributeError:
        config = {}

    # Add console arguments:
    config.update(vars(args))
    return config


def get_parser(*additional_params) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for param in additional_params:
        parser.add_argument('--' + param, '-' + get_short_name(param),
                            dest=param)
    return parser


def get_config(*additional_params) -> dict:
    return parse_config(get_parser(*additional_params))
