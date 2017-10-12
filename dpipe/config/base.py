import functools
import os

from dpipe.externals.resource_manager.resource_manager import ResourceManager, \
    get_module, generate_config

DB_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(DB_DIR, 'modules_db.json')
MODULES_FOLDER = os.path.abspath(os.path.join(DB_DIR, os.pardir))
EXCLUDED_PATHS = ['externals', 'config', 'medim']

get_module = functools.partial(get_module, db_path=DB_PATH)


def get_resource_manager(config_path, **additional) -> ResourceManager:
    rm = ResourceManager(config_path, get_module=get_module)
    # Additional arguments:
    rm.set('config_path', config_path)
    for key, value in additional.items():
        rm.set(key, value, override=True)

    generate_config(MODULES_FOLDER, DB_PATH, 'dpipe', EXCLUDED_PATHS)
    return rm


if __name__ == '__main__':
    generate_config(MODULES_FOLDER, DB_PATH, 'dpipe', EXCLUDED_PATHS)
