import argparse

from dpipe.config import get_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--config_path')
    args = parser.parse_known_args()[0]

    getattr(get_resource_manager(args.config_path), args.command)
