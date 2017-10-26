import importlib

from dpipe.config import register


@register('import', 'meta')
def importer(module: str):
    return importlib.import_module(module)
