from functools import partial


def config_partial(config, module_type, module_builders, **kwargs):
    module_name = config[module_type]
    try:
        module_builder = module_builders[module_name]
    except KeyError:
        raise ValueError(
            'Wrong module name provided\n' +
            'Provided name: {}\n'.format(module_name) +
            'Available names: {}\n'.format([*module_builders.keys()]))
    params = config.get('{}__params'.format(module_type), {})
    return partial(module_builder, **params, **kwargs)


def config_object(config, module_type, module_builders, **kwargs):
    return config_partial(config, module_type, module_builders, **kwargs)()
