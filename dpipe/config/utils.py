from functools import partial


def config_partial(module_type, config, module_builders, **kwargs):
    module_name = config[module_type]
    try:
        module_builder = module_builders[module_type][module_name]
    except KeyError:
        raise ValueError(
            'Wrong module name provided\n' +
            'Module type: {}\n'.format(module_type) + \
            'Provided name: {}\n'.format(module_name) + \
            'Available names: {}\n'.format(
                [*module_builders[module_type].keys()]))
    params = config.get('{}__params'.format(module_type), {})
    return partial(module_builder, **params, **kwargs)


def config_object(module_type, config, module_builders, **kwargs):
    return config_partial(module_type, config, module_builders, **kwargs)()