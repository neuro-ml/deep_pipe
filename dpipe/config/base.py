import os
import functools

from dpipe.externals.resource_manager.resource_manager import ResourceManager, \
    get_resource, generate_config

DB_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(DB_DIR, 'modules_db.json')
MODULES_FOLDER = os.path.abspath(os.path.join(DB_DIR, os.pardir))
EXCLUDED_PATHS = ['externals', 'config']

get_resource = functools.partial(get_resource, db_path=DB_PATH)


def get_resource_manager(config) -> ResourceManager:
    generate_config(MODULES_FOLDER, DB_PATH, 'dpipe', EXCLUDED_PATHS)
    return ResourceManager(config, get_resource=get_resource)


if __name__ == '__main__':
    generate_config(MODULES_FOLDER, DB_PATH, 'dpipe', EXCLUDED_PATHS)
