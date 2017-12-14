import numpy as np
from sklearn.model_selection import KFold, train_test_split

from dpipe.dataset import Dataset


def extract(l, ids):
    return [l[i] for i in ids]


def cv_111(dataset, *, val_size, n_splits):
    """
    Splits the dataset's ids into triplets (train, validation, test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to `val_size`.

    Parameters
    ----------
    dataset: DataSet
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
    ids = dataset.ids
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

    train_val_test_ids = []
    for i, (train_val_indices, test_indices) in enumerate(cv.split(ids)):
        train_val_ids = extract(ids, train_val_indices)
        test_ids = extract(ids, test_indices)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size, random_state=25 + i
        )
        train_val_test_ids.append((train_ids, val_ids, test_ids))

    return train_val_test_ids


def group_train_test_split(x, groups, *, train_size=None, test_size=None,
                           random_state=None, shuffle=True):
    train_groups, test_groups = train_test_split(
        np.unique(groups), train_size=train_size, test_size=test_size,
        random_state=random_state, shuffle=shuffle
    )
    return ([o for i, o in enumerate(x) if groups[i] in train_groups],
            [o for i, o in enumerate(x) if groups[i] in test_groups])


class ShuffleGroupKFold(KFold):
    def split(self, groups):
        names_unique = np.unique(groups)

        for train, test in super().split(names_unique, names_unique):
            train = np.in1d(groups, names_unique[train])
            test = np.in1d(groups, names_unique[test])

            yield np.where(train)[0], np.where(test)[0]


def group_cv_111(dataset: Dataset, *, val_size, n_splits):
    """
    Splits the dataset's ids into triplets (train, validation, test) keeping all the objects
    from a group in the same set (either train, validation or test).
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to `val_size`.

    The splitter guarantees that no objects belonging to the same group will en up in different sets.

    Parameters
    ----------
    dataset: DataSet
        dataset that has a `group` property
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
    ids = dataset.ids
    groups = dataset.groups
    cv = ShuffleGroupKFold(n_splits=n_splits, shuffle=True, random_state=17)

    train_val_test_ids = []
    for i, (train_val_indices, test_indices) in enumerate(cv.split(groups)):
        train_ids, val_ids = group_train_test_split(
            extract(ids, train_val_indices), extract(groups, train_val_indices),
            test_size=val_size, shuffle=True, random_state=25 + i
        )
        train_val_test_ids.append((train_ids, val_ids,
                                   extract(ids, test_indices)))

    return train_val_test_ids


def group_cv_111_pure_011(dataset: Dataset, *, val_size, n_splits):
    """
    Splits the dataset's ids into triplets (train, validation, test) keeping all the objects
    from a group in the same set (either train, validation or test).
    The splitter additionally discards all but one object from each group in the
    validation and test sets.
    The test ids are determined as in the standard K-fold cross-validation setting:
    for each fold a different portion of 1/K ids is kept for testing.
    The remaining (K-1)/K ids are split into train and validation sets according to `val_size`.

    The splitter guarantees that no objects belonging to the same group will en up in different sets.

    Parameters
    ----------
    dataset: DataSet
        dataset that has a `group` property
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

    def _extract_pure(ids):
        return list(filter(lambda x: '^' not in x, ids))

    split = group_cv_111(dataset=dataset, val_size=val_size, n_splits=n_splits)

    return [(train, _extract_pure(val), _extract_pure(test))
            for train, val, test in split]
