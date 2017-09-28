import inspect
import functools

from .utils import *

__all__ = ['register', 'register_inline', 'generate_config', 'get_resource']


class RegistrationSystem:
    def __init__(self):
        self._registry = {}
        self._paths = {}

    def _register(self, resource, module_name, module_type):
        assert (type(module_name) is str or module_name is None and
                type(module_type) is str or module_type is None)

        if module_name is None:
            try:
                module_name = snake_case(resource.__name__)
            except AttributeError:
                # TODO: maybe provide some additional info
                raise TypeError('Could not infer the resource name. Please specify it explicitly.') from None

        stack = inspect.stack()
        source = inspect.getframeinfo(stack[2][0]).filename
        source = os.path.realpath(source)

        if module_type is None:
            module_type = os.path.basename(os.path.dirname(source))

        modules = self._registry.setdefault(module_type, {})
        if module_name in modules and modules[module_name] != resource:
            raise ValueError(f'Trying to register another resource with the same name and type: '
                             f'{module_type}.{module_name}')
        modules[module_name] = resource
        self._paths.setdefault(source, set()).add((module_type, module_name))

        return resource

    def _analyze_file(self, path, source, config):
        config = [x for x in config if x['source'] != source]
        path = os.path.realpath(path)
        importlib.import_module(source)
        modules = self._paths.get(path, [])

        for module_type, module_name in modules:
            config.append({
                'module_type': module_type,
                'module_name': module_name,
                'source_path': path,
                'source': source,
            })

        return config

    def register(self, module_name: str = None, module_type: str = None):
        """
        A class/function decorator that registers a resource.

        Parameters
        ----------
        module_name: str, optional.
            The name of the module. If None, the __name__ attribute converted to snake_case is used.
        module_type: str, optional
            The type of the module. If None, the folder name it lies in is used.
        """

        def decorator(resource):
            return self._register(resource, module_name, module_type)

        return decorator

    def register_inline(self, resource, module_name: str = None, module_type: str = None):
        """
        Registers a resource. For more details refer to the `register` decorator.
        """

        return self._register(resource, module_name, module_type)

    def generate_config(self, root, db_path, upper_module, exclude=None):
        if exclude is None:
            exclude = []
        old_config, old_hashes = read_config(db_path)
        exclude = [os.path.abspath(os.path.join(root, x)) for x in exclude]
        sources_list = walk(root, upper_module, exclude)

        # cleaning up the deleted files
        current_paths = [os.path.realpath(x[0]) for x in sources_list]
        config = []
        for entry in old_config:
            source_path = entry['source_path']
            if source_path in current_paths:
                config.append(entry)
            else:
                # the file is deleted
                old_hashes.pop(source_path, None)

        hashes = {}
        for path, source in sources_list:
            hashes[path] = new_hash = get_hash(path)
            old_hash = old_hashes.get(path, '')
            if old_hash != new_hash:
                config = self._analyze_file(path, source, config)

        with open(db_path, 'w') as file:
            json.dump({'config': config, 'hashes': hashes}, file, indent=2)


registration_system = RegistrationSystem()
register = registration_system.register
register_inline = registration_system.register_inline
generate_config = registration_system.generate_config
