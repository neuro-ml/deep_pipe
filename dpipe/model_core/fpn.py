import torch.nn as nn

from .layers import make_blocks


def build_fpn(structure, make_block, make_up, make_down, split_merge):
    line, *down_structure = structure
    if len(down_structure) == 0:
        assert len(line) == 1, 'f{line}'
        return make_blocks(line[0], make_block)
    else:
        assert len(line) == 3, f'{line}'
        inner_path = line[1] if isinstance(line[1], nn.Module) else make_blocks(line[1], make_block)
        down_path = nn.Sequential(make_down(), *build_fpn(down_structure, make_block, make_up, make_down, split_merge),
                                  make_up())
        return nn.Sequential(*make_blocks(line[0], make_block),
                             split_merge(down_path, inner_path),
                             *make_blocks(line[2], make_block))
