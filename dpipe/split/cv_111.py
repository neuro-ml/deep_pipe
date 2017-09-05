import numpy as np
from sklearn.model_selection import KFold, train_test_split

from dpipe.dataset import Dataset


def get_cv_111(dataset: Dataset, *, val_size, n_splits):
    ids = dataset.patient_ids
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

    train_val_test = []
    for train, test in cv.split(ids):
        train = [ids[i] for i in train]
        test = [ids[i] for i in test]
        train, val = train_test_split(train, test_size=val_size,
                                      random_state=25)
        train_val_test.append((list(train), list(val), list(test)))

    return train_val_test


class ShuffleGroupKFold(KFold):
    def split(self, groups):
        names_unique = np.unique(groups)

        for train, test in super().split(names_unique, names_unique):
            train = np.in1d(groups, names_unique[train])
            test = np.in1d(groups, names_unique[test])

            yield np.where(train)[0], np.where(test)[0]


def get_group_cv_111(dataset: Dataset, *, val_part, n_splits):
    """
    In order to use this splitter, your Dataset needs to have
     a 'groups' property.
    """
    ids = dataset.patient_ids
    groups = dataset.groups
    cv = ShuffleGroupKFold(n_splits=n_splits, shuffle=True, random_state=17)

    train_val_test = []
    for train, test in cv.split(groups):
        train_groups = [groups[i] for i in train]
        # in validation and train we also need non-overlapping groups
        val_groups = int(1 / val_part + .5)
        spl = ShuffleGroupKFold(n_splits=val_groups, shuffle=True,
                                random_state=25)
        train_, val_ = next(spl.split(train_groups))

        train = [ids[i] for i in train_]
        val = [ids[i] for i in val_]
        test = [ids[i] for i in test]
        train_val_test.append((train, val, test))

    return train_val_test


def _extract_pure(ids):
    return list(filter(ids, lambda x: '^' not in x))


def get_pure_val_test_group_cv_111(dataset: Dataset, *, val_part, n_splits):
    """
    In order to use this splitter, your Dataset needs to have
     a 'groups' property.
    """
    split = get_group_cv_111(dataset=dataset, val_part=val_part,
                             n_splits=n_splits)

    return [(train, _extract_pure(val), _extract_pure(test))
            for train, val, test in split]

