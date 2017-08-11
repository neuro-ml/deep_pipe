import inspect
import functools
import importlib


modules_with_folder = [
    'dataset', 'batch_iter', 'experiment', 'model_core', 'split', 'train'
]

module_type2path = {
    'batch_iter_factory': 'utils'
}


def get_module_builders(module_type):
    if module_type in modules_with_folder:
        return importlib.import_module(
            f'dpipe.{module_type}.config').module_builders
    elif module_type in module_type2path:
        path = module_type2path[module_type]
        config = importlib.import_module(f'dpipe.{path}.config')
        return config.__dict__[f'name2{module_type}']
    else:
        raise ValueError("Could't locate builders for "
                         f"module type = {module_type}")


def config_object(config, module_type, lookup_module_type=None, **kwargs):
    return config_partial(config, module_type, lookup_module_type, **kwargs)()


def config_partial(config, module_type, lookup_module_type=None, **kwargs):
    if lookup_module_type is None:
        lookup_module_type = module_type

    module_name = config[lookup_module_type]
    module_builders = get_module_builders(module_type)
    try:
        module_builder = module_builders[module_name]
    except KeyError:
        raise ValueError(
            'Wrong module name provided\n' +
            'Provided name: {}\n'.format(module_name) +
            'Available names: {}\n'.format([*module_builders.keys()]))
    params = config.get('{}__params'.format(lookup_module_type), {})
    return maybe_partial(module_builder, **params, **kwargs)


def maybe_partial(f, **parameters):
    """Function very similar too functools.partial, but it use only those
    parameters that exists in function signature"""
    all_parameters = inspect.signature(f).parameters
    matched_parameters = {}
    for parameter in parameters:
        if parameter in all_parameters:
            default = all_parameters[parameter].default
            if default is inspect.Parameter.empty:
                matched_parameters[parameter] = parameters[parameter]
            else:
                raise ValueError('Attempted to set parameter '
                                 f'{parameter}={parameters[parameter]} '
                                 f'for {f}, which already have'
                                 f'{parameter}={default}')
    return functools.partial(f, **matched_parameters)
