import argparse

__all__ = ['get_parser', 'get_args', 'parse_args']


class ConsoleArguments:
    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]

        console_argument = re.compile(r'^--[^\d\W]\w*$')

        self.args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not console_argument.match(arg):
                raise ValueError(f'Invalid console argument: {arg}')
            try:
                value = int(value)
            except ValueError:
                pass
            self.args[arg] = value

    def __getattr__(self, name):
        try:
            return self.args[name]
        except KeyError:
            raise AttributeError(f'Console argument {name} not provided') from None


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
