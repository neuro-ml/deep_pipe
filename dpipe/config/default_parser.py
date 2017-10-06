import argparse

__all__ = ['get_parser', 'get_args', 'parse_args']


def get_short_name(name: str) -> str:
    """long_informative_name -> lip"""
    return ''.join(map(lambda x: x[0], name.split('_')))


def get_parser(*additional_params) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for param in additional_params:
        parser.add_argument('--' + param, '-' + get_short_name(param),
                            dest=param)
    return parser


def parse_args(parser: argparse.ArgumentParser) -> dict:
    args = parser.parse_args()

    return vars(args)


def get_args(*additional_params) -> dict:
    return parse_args(get_parser(*additional_params))
