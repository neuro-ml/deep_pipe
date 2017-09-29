import json
from dpipe.config import register


@register(module_type='meta')
def extractor(module, *, property):
    return getattr(module, property)


@register(module_type='meta')
def from_json(path):
    with open(path, 'r') as f:
        return json.load(f)
