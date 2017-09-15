import os
import json
import inspect
import functools
import importlib

__all__ = ['register', 'bind_module']

CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'generated_config.json')
MODULES_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))

EXCLUDED_PATHS = ['externals']
EXCLUDED_PATHS = [os.path.join(MODULES_FOLDER, x) for x in EXCLUDED_PATHS]


def register(resource=None, module=None):
    stack = inspect.stack()
    source = inspect.getframeinfo(stack[1][0]).filename
    source = os.path.realpath(source)

    def decorator(entity):
        entity.__module_name__ = module
        entity.__resource_name__ = resource
        entity.__source_path__ = source
        return entity

    return decorator


def bind_module(module):
    return functools.partial(register, module=module)


def analyze_file(path, source, config):
    config = [x for x in config if x['source'] != source]
    imported = importlib.import_module(source)
    real_path = os.path.realpath(path)

    for var in dir(imported):
        variable = getattr(imported, var)
        try:
            module = getattr(variable, '__module_name__')
            resource = getattr(variable, '__resource_name__')
            source_path = getattr(variable, '__source_path__')

            if module is None:
                module = os.path.basename(os.path.dirname(source_path))
            if resource is None:
                # TODO: maybe https://stackoverflow.com/a/1176023
                resource = var

            if source_path == real_path:
                for x in config:
                    if x['module'] == module and x['resource'] == resource:
                        raise ValueError(
                            f'The resource {resource} already exists '
                            f'in module {module}')

                config.append({
                    'source': source,
                    'module': module,
                    'resource': resource,
                    'variable': var,
                    # TODO: write relative path
                    'source_path': source_path,
                })

        except AttributeError:
            pass

    return config


def walk(path, source):
    modules = []
    if path in EXCLUDED_PATHS:
        return modules

    for root, dirs, files in os.walk(path):
        for directory in dirs:
            if not directory.startswith('__'):
                dir_path = os.path.join(root, directory)
                #                 TODO: inspect.getmodulename
                modules.extend(walk(dir_path, f'{source}.{directory}'))

        for file in files:
            name, ext = os.path.splitext(file)
            if not file.startswith(('__', 'test')) and ext == '.py':
                file_path = os.path.join(root, file)
                modules.append((file_path, f'{source}.{name}'))
        break
    return modules


def generate_config(root=MODULES_FOLDER, config_path=CONFIG_PATH,
                    upper_module='dpipe'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        last_changed = os.path.getmtime(config_path)
    except FileNotFoundError:
        config = []
        last_changed = 0

    # TODO: what about renamed?
    # cleaning up the deleted files
    sources_list = walk(root, upper_module)
    current_paths = [os.path.realpath(x[0]) for x in sources_list]
    config = [x for x in config if x['source_path'] in current_paths]

    for path, source in sources_list:
        # TODO: probably keep these times explicitly
        if last_changed >= os.path.getmtime(path):
            continue

        config = analyze_file(path, source, config)

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=2)


if __name__ == '__main__':
    generate_config()
