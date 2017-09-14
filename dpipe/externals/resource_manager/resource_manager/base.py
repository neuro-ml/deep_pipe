import functools


def is_module_definition(definition):
    return (type(definition) is dict and 'name' in definition and
            'type' in definition)


def config_module_partial(module_name, module_type, get_module_builders,
                          **params):
    module_builders = get_module_builders(module_type)
    try:
        module_builder: dict = module_builders[module_name]
    except KeyError:
        raise ValueError(f'Wrong module name "{module_name}" provided\n'
                         f'Available names: {module_builders.keys()}\n')
    return functools.partial(module_builder, **params)


def config_module(module_name, module_type, get_module_builders, **params):
    return config_module_partial(module_name, module_type,
                                 get_module_builders, **params)()


class ResourceManager:
    def __init__(self, config, get_module_builders):
        self.config = config
        self.get_module_builders = get_module_builders
        self.resources = {}

    def __getitem__(self, item):
        return self._request_resource(item)

    def __setitem__(self, key, value):
        assert key not in self.resources, f'Tried to overwrite resource {key}'
        self.resources[key] = value

    def get(self, item, default=None):
        return self.resources.get(item, default)

    def _request_resource(self, name):
        if name not in self.resources:
            try:
                definition = self.config[name]
            except KeyError as e:
                raise TypeError(
                    f"Couldn't find definition for resource {name}\n"
                    f"Config: {self.config}") from e

            self.resources[name] = self._define_resource(definition)
        return self.resources[name]

    def _define_resource(self, definition):
        if is_module_definition(definition):
            return self._define_module(definition)
        else:
            # Consider definition to be a simple python type
            return definition

    def _define_module(self, definition: dict):
        module_name = definition['name']
        module_type = definition['type']
        inputs = self._get_inputs(definition.get('inputs', {}))
        params = definition.get('params', {})
        if definition.get('init', True):
            initialize = config_module
        else:
            initialize = config_module_partial
        return initialize(module_name, module_type, self.get_module_builders,
                          **params, **inputs)

    def _get_resource(self, resource):
        if type(resource) is str:
            return self._request_resource(resource)
        elif type(resource) is list:
            return [self._get_resource(r) for r in resource]
        elif is_module_definition(resource):
            return self._define_module(resource)
        else:
            raise ValueError("Couldn't get resource:\n{resource}\nUnknown type")

    def _get_inputs(self, inputs_definition):
        return {name: self._get_resource(resource)
                for name, resource in inputs_definition.items()}
