import argparse

import lazycon
from pathlib import Path

def build():
    parser = argparse.ArgumentParser('Build an experiment layout from the provided config.', add_help=False)
    parser.add_argument('config')
    args = parser.parse_known_args()[0]

    layout = lazycon.load(args.config).layout
    layout.build_parser(parser)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this message and exit')

    args = parser.parse_args()
    layout.build(**vars(args))


def run():
    parser = argparse.ArgumentParser('Run an experiment based on the provided config.', add_help=False)
    parser.add_argument('config')
    args = parser.parse_known_args()[0]
    config_path = Path(args.config).absolute()

    layout = lazycon.load(config_path).layout
    layout.run_parser(parser)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this message and exit')

    args = parser.parse_args()
    folds = args.folds
    layout.run(config=config_path, folds=folds)
