from experiments.batch_iterators.patch_3d_stratified import make_batch_iter
from experiments.datasets import config_dataset


if __name__ == '__main__':
    dataset = config_dataset('brats2017')
    patients = list(dataset.patient_ids[:30]) * 100
    batch_iter = make_batch_iter(
        patients, dataset, batch_size=20, x_patch_sizes=[[25, 25, 25]],
        y_patch_size=[15, 15, 15], nonzero_fraction=0.5)

    with batch_iter:
        for x_batch, y_batch in batch_iter:
            print(y_batch.mean(), flush=True)
