import argparse
from pathlib import Path

from resource_manager import read_config, ResourceManager

MODULES_FOLDER = Path(__file__).resolve().parent.parent
SHORTCUTS = {
    'dpipe_configs': MODULES_FOLDER.parent / 'dpipe_configs',
}


def get_resource_manager(source_path: str, shortcuts: dict = None, injections: dict = None) -> ResourceManager:
    """Read and parse a config. See ``resource_manager.read_config`` for details."""
    return read_config(source_path, shortcuts={**SHORTCUTS, **(shortcuts or {})}, injections=injections)


def render_config_resource():
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('-cp', '--config_path', required=True)
    args = parser.parse_known_args()[0]

    get_resource_manager(args.config_path).get_resource(args.command)
