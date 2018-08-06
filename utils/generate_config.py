import argparse

from dpipe.config import get_resource_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('result_path')
    args = parser.parse_args()
    get_resource_manager(args.config_path).save_config(args.result_path)
