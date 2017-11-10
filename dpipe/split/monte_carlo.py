from sklearn.model_selection import train_test_split
from dpipe.dataset import Dataset
from dpipe.config import register

@register()
def monte_carlo(dataset: Dataset, *, train_fraction, val_fraction, n_splits):
    """
    Example: train_fraction = 0.8, val_fraction = 0.1, n_splits = 2 will create 2 splits such that 80% of ids are in
    the train sample, 10% - in val sample, 10% - in test sample.
    """
    if not (train_fraction > 0 and val_fraction >= 0 and train_fraction + val_fraction <= 1):
        raise ValueError

    ids = dataset.patient_ids
    train_val_test_ids = []

    random_state = 52
    for i in range(n_splits):
        train_val_ids, test_ids = train_test_split(
            ids, train_size=train_fraction + val_fraction, random_state=random_state + i
        )
        train_fraction_normalized = train_fraction / (train_fraction + val_fraction)
        train_ids, val_ids = train_test_split(
            train_val_ids, train_size=train_fraction_normalized, random_state=random_state + i
        )
        train_val_test_ids.append((train_ids, val_ids, test_ids))

    return train_val_test_ids
