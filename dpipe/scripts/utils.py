def read_lines(path):
    with open(path, 'r') as file:
        return [l for l in map(lambda x: x.strip(), file) if l != '']
