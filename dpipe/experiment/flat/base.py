import os
import json
import shutil

from dpipe.config import register, get_resource_manager


@register('flat', 'experiment')
def build_flat_structure(split, config_path, experiment_path, *, makefile):
    makefile_path = os.path.join(os.path.dirname(__file__), makefile)
    assert os.path.exists(makefile_path), f'no {makefile_path} found'

    for i, ids in enumerate(split):
        local = os.path.join(experiment_path, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(makefile_path, os.path.join(local, 'Snakefile'))

        for val, prefix in zip(ids, ('train', 'val', 'test')):
            path = os.path.join(local, f'{prefix}_ids.json')
            with open(path, "w") as f:
                json.dump(val, f, indent=0)

    # resource manager is needed here, because there may be inheritance
    rm = get_resource_manager(config_path)
    rm.save_config(os.path.join(experiment_path, 'config'))
