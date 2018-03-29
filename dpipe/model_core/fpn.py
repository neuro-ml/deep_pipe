import torch.nn as nn

from .layers import compose_blocks


def build_fpn(structure, cb, up, down, split_merge):
    line, *down_structure = structure
    if len(down_structure) == 0:
        assert len(line) == 1, 'f{line}'
        return compose_blocks(line[0], cb)
    else:
        assert len(line) == 3, f'{line}'
        inner_path = line[1] if isinstance(line[1], nn.Module) else compose_blocks(line[1], cb)
        down_path = nn.Sequential(down(), *build_fpn(down_structure, cb, up, down, split_merge), up())
        return nn.Sequential(*compose_blocks(line[0], cb),
                             split_merge(down_path, inner_path),
                             *compose_blocks(line[2], cb))
