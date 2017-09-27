import inspect
import functools

from .utils import *

__all__ = ['register', 'bind_module', 'register_inline', 'generate_config', 'get_resource']


def register(module_name: str = None, module_type: str = None):
    """
    A class/function decorator that registers a resource.

    Parameters
    ----------
    module_name: str, optional.
        The name of the module. If None, the variable name converted to snake_case is used.
    module_type: str, optional
        The type of the module. If None, the folder name it lies in is used.
    """
    assert (type(module_name) is str or module_name is None and
            type(module_type) is str or module_type is None)

    stack = inspect.stack()
    source = inspect.getframeinfo(stack[1][0]).filename
    source = os.path.realpath(source)

    def decorator(entity):
        entity.__module_name__ = module_name
        entity.__module_type__ = module_type
        entity.__source_path__ = source
        return entity

    return decorator


def bind_module(module_type):
    """
    A factory for decorators with fixed module type.

    Parameters
    ----------
    module_type: str
        The type of the module.

    Returns
    -------

    A class/function decorator with fixed module type
    """
    return functools.partial(register, module_type=module_type)


def register_inline(resource, module_name: str = None, module_type: str = None):
    """
    Registers a resource. For more details refer to the `register` decorator.
    """
    assert (type(module_name) is str or module_name is None and
            type(module_type) is str or module_type is None)

    stack = inspect.stack()
    source = inspect.getframeinfo(stack[1][0]).filename
    source = os.path.realpath(source)

    resource.__module_name__ = module_name
    resource.__module_type__ = module_type
    resource.__source_path__ = source

    return resource


def generate_config(root, db_path, upper_module, exclude):
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
            old_hashes.pop(source_path)

    hashes = {}
    for path, source in sources_list:
        hashes[path] = new_hash = get_hash(path)
        old_hash = old_hashes.get(path, '')
        if old_hash != new_hash:
            config = analyze_file(path, source, config)

    with open(db_path, 'w') as file:
        json.dump({'config': config, 'hashes': hashes}, file, indent=2)


def get_resource(module_type, module_name, db_path):
    config = read_config(db_path)[0]

    for entry in config:
        try:
            if entry['module_type'] == module_type and \
                            entry['module_name'] == module_name:
                source = importlib.import_module(entry['source'])
                return getattr(source, entry['variable'])
        except (AttributeError, KeyError):
            handle_corruption()

    raise KeyError(f'The module "{module_name}" of type "{module_type}" '
                   f'was not found')
