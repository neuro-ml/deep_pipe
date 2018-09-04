import os
import argparse

from resource_manager import read_config, ResourceManager

MODULES_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SHORTCUTS = {
    'dpipe_configs': os.path.join(MODULES_FOLDER, os.pardir, 'dpipe_configs'),
}


def get_resource_manager(source_path: str, shortcuts: dict = None) -> ResourceManager:
    """Read and parse a config. See `resource_manager.read_config for details.`"""
    return read_config(source_path, shortcuts={**SHORTCUTS, **(shortcuts or {})})


def render_config_resource():
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--config_path', required=True)
    args = parser.parse_known_args()[0]

    get_resource_manager(args.config_path).get_resource(args.command)
