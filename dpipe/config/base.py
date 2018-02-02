import os

from resource_manager import ResourceManager, get_module as _get_module, generate_config

DB_DIR = os.path.abspath(os.path.dirname(__file__))
MODULES_FOLDER = os.path.abspath(os.path.join(DB_DIR, os.pardir))

MODULES_DB = os.path.join(DB_DIR, 'modules_db.json')
EXTERNALS = os.path.join(MODULES_FOLDER, 'externals')
USER = os.path.expanduser('~')
RC = os.path.expanduser('~/.dpiperc')
SHORTCUTS = {
    'dpipe_configs': os.path.join(MODULES_FOLDER, os.pardir, 'config_examples'),
    'config_examples': os.path.join(MODULES_FOLDER, os.pardir, 'config_examples')
}

_modules_were_generated = False


def get_module(module_type, module_name):
    global _modules_were_generated
    if not _modules_were_generated:
        generate_config(EXTERNALS, MODULES_DB, 'dpipe.externals')
        _modules_were_generated = True
    return _get_module(module_type, module_name, db_path=MODULES_DB)


def link_externals():
    try:
        with open(RC) as file:
            externals = file.read().split()
    except FileNotFoundError:
        externals = []
    externals = {os.path.realpath(os.path.join(USER, os.path.expanduser(x))) for x in externals}
    modules = {os.path.basename(x) for x in externals}
    if len(modules) < len(externals):
        # TODO: provide workaround
        raise ValueError('There are modules with duplicate names in ~/.dpiperc')

    os.makedirs(EXTERNALS, exist_ok=True)
    linked = [os.path.join(EXTERNALS, x) for x in os.listdir(EXTERNALS)]
    linked = {os.path.realpath(x): x for x in linked}

    for path in set(linked) - externals:
        os.unlink(linked[path])

    for path in externals - set(linked):
        name = os.path.basename(path)
        os.symlink(path, os.path.join(EXTERNALS, name))


def get_resource_manager(config_path: str) -> ResourceManager:
    """
    Get the ResourceManager corresponding to the config from `config_path`.

    Parameters
    ----------
    config_path: str
        path to the config to parse

    Returns
    -------
    resource_manager: ResourceManager
    """
    link_externals()
    return ResourceManager.read_config(config_path, get_module=get_module, shortcuts=SHORTCUTS)


if __name__ == '__main__':
    link_externals()
