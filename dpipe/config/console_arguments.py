import argparse
import re

__all__ = ['get_parser', 'get_args', 'parse_args']

from resource_manager import register


@register('console', 'meta')
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
