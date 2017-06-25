import gc
from random import choice

from tqdm import tqdm

from experiments.batch_iterators import make_3d_patch_stratified_iter
from experiments.datasets import config_dataset, make_cached
from experiments.dl import Model, ModelController
from experiments.dl import Optimizer
from experiments.dl.models.deepmedic_orig import DeepMedic
from experiments.splits import make_cv_111


n_epoch = 2
val_size = 6
n_splits = 100
n_iter_per_epoch = 2


def extract(x, idx):
    return [x[i] for i in idx]

if __name__ == '__main__':
    dataset = make_cached(config_dataset('brats2017'))

    tran_val_test = make_cv_111(val_size=val_size, n_splits=n_splits)(dataset)

    optimizer = Optimizer()
    model = DeepMedic(optimizer, dataset.n_chans_mscan,
                      dataset.n_chans_msegm, n_parts=[1, 2, 1])

    for i, (train, val, test) in tqdm(enumerate(tran_val_test)):
        log_path = f'logs/deepmedic/{i}'
        train_patient_ids = extract(dataset.patient_ids, train)

        model_controller = ModelController(model, log_path)

        with model_controller:
            for i in range(n_epoch):
                batch_iter = make_3d_patch_stratified_iter(
                    train_patient_ids, dataset, batch_size=64,
                    x_patch_sizes=[[25, 25, 25], [57, 57, 57]],
                    y_patch_size=[9, 9, 9], nonzero_fraction=0.5)

                with batch_iter:
                    model_controller.train(batch_iter, 0.1, n_iter_per_epoch)
        print(gc.collect())


