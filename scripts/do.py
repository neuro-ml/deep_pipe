import argparse

from dpipe.config import get_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--config_path')
    args = parser.parse_known_args()[0]

    # Here it is important that we don't save resource manager. If it was saved in the global scope, it won't be
    # deleted until all children threads are dead. But some threads (from infinite batch iterators) from pdp module
    # will get closed only when corresponding batch iterator will be deleted. Therefore, it's a deadlock.
    # So, we have to delete it manually or don't save it at all.
    getattr(get_resource_manager(args.config_path), args.command)
