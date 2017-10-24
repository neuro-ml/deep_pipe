import argparse
import json
import re

from dpipe.config import register


@register(module_name='json', module_type='io')
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


console_argument = re.compile(r'^--[^\d\W]\w*$')


@register('console', 'io')
class ConsoleArguments:
    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]

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
