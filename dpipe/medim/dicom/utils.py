import pandas as pd

ORIENTATION = [f'ImageOrientationPatient{i}' for i in range(6)]


def split_floats(string):
    return list(map(float, string.split(',')))


def contains_info(row, *cols):
    return all(col in row and pd.notnull(row[col]) for col in cols)
