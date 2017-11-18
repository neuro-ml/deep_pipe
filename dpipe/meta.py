import importlib

from dpipe.config import register


@register('import', 'meta')
def importer(module: str):
    return importlib.import_module(module)


@register(module_type='meta')
def compose(functions: list):
    def wrapped(arg):
        for func in functions:
            arg = func(arg)
        return arg

    return wrapped
