import json
import os

from resource_manager import ResourceManager, get_module as _get_module, generate_config

DB_DIR = os.path.abspath(os.path.dirname(__file__))
MODULES_FOLDER = os.path.abspath(os.path.join(DB_DIR, os.pardir))

MODULES_DB = os.path.join(DB_DIR, 'modules_db.json')
EXTERNALS_DB = os.path.join(DB_DIR, 'externals_db.json')
EXTERNALS = os.path.join(MODULES_FOLDER, 'externally_loaded_resources')
USER = os.path.expanduser('~')
RC = os.path.expanduser('~/.dpiperc')

_modules_were_generated = False


def get_module(module_type, module_name):
    global _modules_were_generated
    if not _modules_were_generated:
        generate_config(EXTERNALS, MODULES_DB, 'dpipe.externally_loaded_resources')
        _modules_were_generated = True
    return _get_module(module_type, module_name, db_path=MODULES_DB)


def load(path, default):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return default


def link_externals():
    os.makedirs(EXTERNALS, exist_ok=True)
    db = json.loads(load(EXTERNALS_DB, '{}'))
    if set(db.values()) ^ set(os.listdir(EXTERNALS)):
        raise OSError('"externally_loaded_resources" not clean. Were some resources manually linked?')

    loaded = set(db.keys())
    to_load = {os.path.join(USER, x) for x in load(RC, '').split()}
    for path in loaded - to_load:
        os.unlink(os.path.join(EXTERNALS, db[path]))
        db.pop(path)

    for path in to_load - loaded:
        i = 0
        while f'module{i}' in db.values():
            i += 1
        name = f'module{i}'
        os.symlink(path, os.path.join(EXTERNALS, name))
        db[path] = name

    with open(EXTERNALS_DB, 'w') as f:
        json.dump(db, f)


def get_resource_manager(config_path) -> ResourceManager:
    link_externals()
    rm = ResourceManager(config_path, get_module=get_module)
    if 'torch' in rm._get_whole_config():
        import torch
    return rm
