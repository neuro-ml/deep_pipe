import numpy as np
from sklearn.model_selection import KFold, train_test_split

from dpipe.config import register
from dpipe.dataset import Dataset


def extract(l, ids):
    return [l[i] for i in ids]


@register()
def cv_111(dataset: Dataset, *, val_size, n_splits):
    ids = dataset.patient_ids
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


@register()
def group_cv_111(dataset: Dataset, *, val_size, n_splits):
    """
    In order to use this splitter, your Dataset needs to have
     a 'groups' property.
    """
    ids = dataset.patient_ids
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


@register()
def group_cv_111_pure_011(dataset: Dataset, *, val_size, n_splits):
    """
    In order to use this splitter, your Dataset needs to have
     a 'groups' property.
    """

    def _extract_pure(ids):
        return list(filter(lambda x: '^' not in x, ids))

    split = group_cv_111(dataset=dataset, val_size=val_size, n_splits=n_splits)

    return [(train, _extract_pure(val), _extract_pure(test))
            for train, val, test in split]
