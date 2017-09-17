import inspect
import functools

from .utils import *

__all__ = ['register', 'bind_module', 'generate_config', 'get_resource']


def register(module_name: str = None, module_type: str = None):
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
    return functools.partial(register, module_type=module_type)


def generate_config(root, config_path, upper_module, exclude):
    old_config, old_times, last_changed = read_config(config_path)
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
            old_times.pop(source_path)

    times = {}
    for path, source in sources_list:
        times[path] = os.path.getmtime(path)
        old_time = old_times.get(path)
        if old_time is not None and last_changed >= old_time:
            continue

        config = analyze_file(path, source, config)

    with open(config_path, 'w') as file:
        json.dump({'config': config, 'times': times}, file, indent=2)


def get_resource(module_type, module_name, config_path):
    config = read_config(config_path)[0]

    for entry in config:
        try:
            if entry['module_type'] == module_type and \
                            entry['module_name'] == module_name:
                source = importlib.import_module(entry['source'])
                return getattr(source, entry['variable'])
        except (AttributeError, KeyError):
            handle_corruption()

    raise KeyError(f'The module "{module_name}" of type "{module_type}"'
                   f'was not found')
