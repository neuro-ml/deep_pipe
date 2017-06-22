from .base import Dataset
from .brats import Brats2015, Brats2017


dataset_name2dataset = {
    'brats2015': Brats2015,
    'brats2017': Brats2017,
}

dataset_name2default_path = {
    'brats2015': '/home/mount/neuro-x02-ssd/brats2015/processed',
    'brats2017': '/home/mount/neuro-x02-ssd/brats2017/processed',
}


def config_dataset(dataset_name: str, dataset_path=None) -> Dataset:
    if dataset_path is None:
        dataset_path = dataset_name2default_path[dataset_name]
    dataset = dataset_name2dataset[dataset_name](dataset_path)
    return dataset