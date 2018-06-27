import numpy as np

from .cv import ShuffleGroupKFold, train_test_split_groups
from sklearn.model_selection import KFold


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
            train_val, val_size=val_size, groups=sub_groups, **kwargs)
        new_splits.append([train, val, test])
    return new_splits


def indices_to_subj_ids(splits, subj_ids):
    """Converts split indices to subject IDs"""
    return [list(map(lambda ids: [subj_ids[i] for i in ids], split)) for split in splits]


def get_loo_cv(ids, groups, *, val_size=None, random_state=42):
    """Leave one group out CV. Validation subset will be selected randomly"""
    n_splits = len(np.unique(groups))
    splits = kfold_split(ids, n_splits, groups=groups)
    if val_size is not None:
        splits = split_train(splits, val_size, random_state=random_state)
    return indices_to_subj_ids(splits, ids)


def get_cv_11(ids, *, n_splits, random_state=42):
    split_indices = kfold_split(ids, n_splits, random_state=random_state)
    return indices_to_subj_ids(split_indices, ids)


def get_cv_111(ids, *, val_size, n_splits, random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to `val_size`.

    Parameters
    ----------
    ids
    val_size: float, int
        If float, should be between 0.0 and 1.0 and represents the proportion
        of the `train set` to include in the validation set. If int, represents the
        absolute number of validation samples.
    n_splits: int
        the number of cross-validation folds

    Returns
    -------
    splits: Sequence of triplets
    """
    split_indices = kfold_split(subj_ids=ids, n_splits=n_splits, random_state=random_state)
    split_indices = split_train(splits=split_indices, val_size=val_size, random_state=random_state)
    return indices_to_subj_ids(splits=split_indices, subj_ids=ids)


def get_group_cv_111(ids, groups: np.array, *, val_size, n_splits, random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test) keeping all the objects
    from a group in the same set (either train, validation or test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to `val_size`.

    The splitter guarantees that no objects belonging to the same group will en up in different sets.

    Parameters
    ----------
    ids
    groups: np.array[int]
    val_size: float, int
        If float, should be between 0.0 and 1.0 and represents the proportion
        of the `train set` to include in the validation set. If int, represents the
        absolute number of validation samples.
    n_splits: int
        the number of cross-validation folds

    Returns
    -------
    splits: Sequence of triplets
    """
    split_indices = kfold_split(ids, n_splits, groups=groups, random_state=random_state)
    split_indices = split_train(split_indices, val_size, groups=groups, random_state=random_state)
    return indices_to_subj_ids(split_indices, ids)
