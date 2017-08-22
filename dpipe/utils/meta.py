import json


def extractor(module, *, property):
    return getattr(module, property)


def from_json(path):
    with open(path, 'r') as f:
        return json.load(f)
