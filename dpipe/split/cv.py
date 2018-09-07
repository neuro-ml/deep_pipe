import numpy as np

from .base import split_train, kfold_split, indices_to_subj_ids


def split(ids, *, n_splits, random_state=42):
    split_indices = kfold_split(ids, n_splits, random_state=random_state)
    return indices_to_subj_ids(split_indices, ids)


def leave_group_out(ids, groups, *, val_size=None, random_state=42):
    """Leave one group out CV. Validation subset will be selected randomly"""
    n_splits = len(np.unique(groups))
    splits = kfold_split(ids, n_splits, groups=groups)
    if val_size is not None:
        splits = split_train(splits, val_size, random_state=random_state)
    return indices_to_subj_ids(splits, ids)


def train_val_test_split(ids, *, val_size, n_splits, random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to ``val_size``.

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


def group_train_val_test_split(ids, groups: np.array, *, val_size, n_splits, random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test) keeping all the objects
    from a group in the same set (either train, validation or test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to ``val_size``.

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
