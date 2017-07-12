def read_lines(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return [l for l in lines if l != '']
