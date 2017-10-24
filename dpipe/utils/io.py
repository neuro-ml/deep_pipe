import json
from dpipe.config import register


@register(module_name='json', module_type='io')
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

