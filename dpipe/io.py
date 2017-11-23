import argparse
import json
import re


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


console_argument = re.compile(r'^--[^\d\W]\w*$')


class ConsoleArguments:
    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]
        # allow for positional arguments:
        while args and not console_argument.match(args[0]):
            args = args[1:]

        self.args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not console_argument.match(arg):
                raise ValueError(f'Invalid console argument: {arg}')
            arg = arg[2:]
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
