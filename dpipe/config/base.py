import os
import functools

from resource_manager import read_config

MODULES_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SHORTCUTS = {
    'config_examples': os.path.join(MODULES_FOLDER, os.pardir, 'config_examples')
}

get_resource_manager = functools.partial(read_config, shortcuts=SHORTCUTS)
