import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, GroupShuffleSplit


class ShuffleGroupKFold(KFold):
    def split(self, *, X, groups):
        names_unique = np.unique(groups)
        for train, test in super().split(names_unique):
            train = np.in1d(groups, names_unique[train])
            test = np.in1d(groups, names_unique[test])
            yield np.where(train)[0], np.where(test)[0]


def train_test_split_groups(X, *, val_size, groups=None, **kwargs):
    split_class = (ShuffleSplit if groups is None else GroupShuffleSplit)
    split = split_class(test_size=val_size, **kwargs)
    train, val = next(split.split(X=X, groups=groups))
    return X[train], X[val]


def kfold_split(subj_ids, n_splits, groups=None, **kwargs):
    """Perform kfold (or group kfold) shuffled split
    Returns indices: [[train_0, val_0], ..., [train_k, val_k]]
    """
    kfoldClass = KFold if groups is None else ShuffleGroupKFold
    kfold = kfoldClass(n_splits, shuffle=True, **kwargs)
    return [split for split in kfold.split(X=subj_ids, groups=groups)]


def split_train(splits, val_size, groups=None, **kwargs):
    """Add additional splits of train subsets to output of kfold
    Returns indices: [[train_0, val_0, test_0], ..., [train_k, val_k, test_k]]
    """
    new_splits = []
    for train_val, test in splits:
        sub_groups = None if groups is None else groups[train_val]
        train, val = train_test_split_groups(
            train_val, val_size=val_size, groups=sub_groups, **kwargs) if val_size > 0 else (train_val, [])
        new_splits.append([train, val, test])
    return new_splits


def indices_to_ids(splits, ids):
    """Converts split indices to subject IDs"""
    return [[[ids[i] for i in ids_group] for ids_group in split] for split in splits]
