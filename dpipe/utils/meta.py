def extractor(module, *, property):
    return getattr(module, property)


def from_json(path):
    with open(path, 'r') as file:
        return [l for l in map(lambda x: x.strip(), file) if l != '']
