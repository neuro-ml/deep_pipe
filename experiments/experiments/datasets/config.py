from .base import Dataset
from .brats import Brats2015, Brats2017
from .isles import siss_factory, spes_factory

dataset_name2dataset = {
    'brats2015': Brats2015,
    'brats2017': Brats2017,

    'isles_siss': siss_factory('siss2015.csv'),
    'isles_siss_augmented': siss_factory('augmented_siss.csv'),
    'isles_spes': spes_factory('spes2015.csv'),
    'isles_spes_augmented_core': spes_factory('augmented_spes_core.csv'),
    'isles_spes_augmented_penumbra': spes_factory('augmented_spes_penumbra.csv'),
}

dataset_name2default_path = {
    'brats2015': '/home/mount/neuro-x02-ssd/brats2015/processed',
    'brats2017': '/home/mount/neuro-x02-ssd/brats2017/processed',

    'isles_siss': '/home/mount/neuro-x04-hdd/ISLES/',
    'isles_siss_augmented': '/home/mount/neuro-x04-hdd/ISLES/',
    'isles_spes': '/home/mount/neuro-x04-hdd/ISLES/',
    'isles_spes_augmented_core': '/home/mount/neuro-x04-hdd/ISLES/',
    'isles_spes_augmented_penumbra': '/home/mount/neuro-x04-hdd/ISLES/',
}


def config_dataset(dataset_name: str, dataset_path=None) -> Dataset:
    if dataset_path is None:
        dataset_path = dataset_name2default_path[dataset_name]
    dataset = dataset_name2dataset[dataset_name](dataset_path)
    return dataset
