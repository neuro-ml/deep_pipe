import os
import argparse
from os.path import join
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from dpipe.io import load_json

sets = ['train', 'test']
metrics = ['dice', 'hmm']


def load_results(experiment_path):
    results = defaultdict(dict)

    for fold_name in os.listdir(experiment_path):
        if fold_name.startswith('experiment'):
            for s in sets:
                for metric in metrics:
                    try:
                        partial_result = load_json(join(experiment_path, fold_name, s + '_metrics', metric + '.json'))
                        if hasattr(partial_result[list(partial_result.keys())[0]], '__iter__'):
                            partial_result = {i: np.array(r).flatten() for i, r in partial_result.items()}
                        results[f'{s}_{metric}'].update(partial_result)
                    except FileNotFoundError:
                        pass

    return results


def gather_results(experiments_path):
    records = {}

    for experiment_name in tqdm(os.listdir(experiments_path)):
        experiment_path = join(experiments_path, experiment_name)
        if not os.path.isdir(experiment_path) or experiment_name.startswith('.'):
            continue

        results = load_results(experiment_path)

        record = {}
        for name, result in results.items():
            result = np.mean(list(result.values()), axis=0)
            if hasattr(result, '__iter__'):
                for i, v in enumerate(result):
                    record[f'{name}_{i}'] = result[i]
            else:
                record[name] = result

        records[experiment_name] = record

    df = pd.DataFrame.from_dict(records, orient='index')
    return df[sorted(df.columns.tolist())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments_path')
    args = parser.parse_known_args()[0]

    df = gather_results(args.experiments_path)
    df.to_csv('results.csv', index_label='experiment_name')
