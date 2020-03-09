import argparse

from resource_manager import read_config


def build():
    parser = argparse.ArgumentParser('Build an experiment layout from the provided config.', add_help=False)
    parser.add_argument('config')
    args = parser.parse_known_args()[0]

    layout = read_config(args.config).layout
    layout.build_parser(parser)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this message and exit')

    args = parser.parse_args()
    layout.build(**vars(args))


def run():
    parser = argparse.ArgumentParser('Run an experiment based on the provided config.', add_help=False)
    parser.add_argument('config')
    args = parser.parse_known_args()[0]

    layout = read_config(args.config).layout
    layout.run_parser(parser)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this message and exit')

    args = parser.parse_args()
    layout.run(**vars(args))
