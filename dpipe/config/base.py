import argparse
from pathlib import Path
import numpy as np

from resource_manager import read_config, ResourceManager

MODULES_FOLDER = Path(__file__).resolve().parent.parent
SHORTCUTS = {
    'dpipe_configs': MODULES_FOLDER.parent / 'dpipe_configs',
}


@np.deprecate(new_name='resource_manager.read_config')
def get_resource_manager(source_path: str, shortcuts: dict = None, injections: dict = None) -> ResourceManager:
    """
    Read and parse a config. See ``resource_manager.read_config`` for details.

    Warnings
    --------
    This function is deprecated. Use ``resource_manager.read_config`` instead:
    >>> from resource_manager import read_config
    >>> config = read_config('some_path.config')
    """
    return read_config(source_path, shortcuts={**SHORTCUTS, **(shortcuts or {})}, injections=injections)


def render_config_resource():
    parser = argparse.ArgumentParser(epilog='This command will soon change its behaviour. Use `run-config` instead.')
    parser.add_argument('command')
    parser.add_argument('-cp', '--config_path', required=True)
    args = parser.parse_known_args()[0]

    read_config(args.config_path).get_resource(args.command)
