import unittest

import numpy as np

from .config import config_dataset, dataset_name2dataset


class TestDatasets(unittest.TestCase):
    def test_datasets(self):
        for dataset_name in dataset_name2dataset:
            dataset = config_dataset()
            patient_ids = dataset.patient_ids
            patient_ids = [patient_ids[0], patient_ids[-1]]

            for p in patient_ids:
                mscan = dataset.load_mscan(p)
                segm = dataset.load_segm(p)
                msegm = dataset.load_msegm(p)

                self.assertEqual(mscan.shape[0], dataset.n_chans_mscan)
                self.assertEqual(msegm.shape[0], dataset.n_chans_msegm)
                self.assertEqual(len(np.unique(segm)), dataset.n_classes)
                self.assertSequenceEqual(mscan.shape[1:], dataset.spatial_size)

                self.assertSequenceEqual(
                    list(dataset.segm2msegm(segm).flatten()),
                    list(msegm.flatten()))
