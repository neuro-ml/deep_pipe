"""Script that runs command from config."""
import argparse

from dpipe.config import get_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--config_path')
    args = parser.parse_known_args()[0]

    get_resource_manager(args.config_path).get_resource(args.command)
