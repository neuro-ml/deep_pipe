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

dataset_name2default_params = {
    'brats2015': {
        'data_path': '/home/mount/neuro-x02-ssd/brats2015/processed'},
    'brats2017': {
        'data_path': '/home/mount/neuro-x02-ssd/brats2017/processed'},

    'isles_siss': {
        'data_path': '/home/mount/neuro-x04-hdd/ISLES/'},
    'isles_siss_augmented': {
        'data_path': '/home/mount/neuro-x04-hdd/ISLES/'},
    'isles_spes': {
        'data_path': '/home/mount/neuro-x04-hdd/ISLES/'},
    'isles_spes_augmented_core': {
        'data_path': '/home/mount/neuro-x04-hdd/ISLES/'},
    'isles_spes_augmented_penumbra': {
        'data_path': '/home/mount/neuro-x04-hdd/ISLES/'},
}
