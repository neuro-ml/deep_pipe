import os
import json
import importlib

from dpipe.externals.resource_manager.resource_manager import ResourceManager

with open(os.path.join(os.path.dirname(__file__),
                       'module_type2path.json')) as f:
    module_type2path = json.load(f)


def get_module_builders(module_type):
    try:
        path = module_type2path[module_type]
    except KeyError as e:
        raise TypeError("Could't locate builders for module type:"
                        f" {module_type}") from e

    config = importlib.import_module(path)
    return getattr(config, f'name2{module_type}')


def get_resource_manager(config) -> ResourceManager:
    return ResourceManager(config, get_module_builders=get_module_builders)
