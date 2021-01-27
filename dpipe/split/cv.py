from typing import Sequence, Union, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from dpipe.itertools import extract
from .base import split_train, kfold_split, indices_to_ids


def split(ids, *, n_splits, random_state=42):
    split_indices = kfold_split(ids, n_splits, random_state=random_state)
    return indices_to_ids(split_indices, ids)


def leave_group_out(ids, groups, *, val_size=None, random_state=42):
    """Leave one group out CV. Validation subset will be selected randomly."""
    n_splits = len(np.unique(groups))
    splits = kfold_split(ids, n_splits, groups=groups)
    if val_size is not None:
        splits = split_train(splits, val_size, random_state=random_state)
    return indices_to_ids(splits, ids)


def train_val_test_split(ids, *, val_size, n_splits, random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K - 1) / K ids are split into train and validation sets according to ``val_size``.

    Parameters
    ----------
    ids
    val_size: float, int
        If ``float``, should be between 0.0 and 1.0 and represents the proportion
        of the train set to include in the validation set. If ``int``, represents the
        absolute number of validation samples.
    n_splits: int
        the number of cross-validation folds.

    Returns
    -------
    splits: Sequence of triplets
    """
    split_indices = kfold_split(subj_ids=ids, n_splits=n_splits, random_state=random_state)
    split_indices = split_train(splits=split_indices, val_size=val_size, random_state=random_state)
    return indices_to_ids(split_indices, ids)


def group_train_val_test_split(ids: Sequence, groups: Union[Callable, Sequence], *, val_size, n_splits,
                               random_state=42):
    """
    Splits the dataset's ids into triplets (train, validation, test) keeping all the objects
    from a group in the same set (either train, validation or test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1 / K ids is kept for testing.
    The remaining (K - 1) / K ids are split into train and validation sets according to ``val_size``.

    The splitter guarantees that no objects belonging to the same group will en up in different sets.

    Parameters
    ----------
    ids
    groups: np.ndarray[int]
    val_size: float, int
        If ``float``, should be between 0.0 and 1.0 and represents the proportion
        of the train set to include in the validation set. If ``int``, represents the
        absolute number of validation samples.
    n_splits: int
        the number of cross-validation folds
    """
    if callable(groups):
        groups = list(map(groups, ids))
    groups = np.asarray(groups)
    split_indices = kfold_split(ids, n_splits, groups=groups, random_state=random_state)
    split_indices = split_train(split_indices, val_size, groups=groups, random_state=random_state)
    return indices_to_ids(split_indices, ids)


def stratified_train_val_test_split(ids: Sequence, labels: Union[Callable, Sequence], *, val_size, n_splits,
                                    random_state=42):
    if callable(labels):
        labels = list(map(labels, ids))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_val_test_ids = []
    for i, (train_val_indices, test_indices) in enumerate(cv.split(ids, labels)):
        train_val_ids = extract(ids, train_val_indices)
        test_ids = extract(ids, test_indices)
        if val_size:
            train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=25 + i)
        else:
            train_ids, val_ids = train_val_ids, []

        train_val_test_ids.append((train_ids, val_ids, test_ids))

    return train_val_test_ids
