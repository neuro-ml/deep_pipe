import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, GroupShuffleSplit
from dpipe.dataset import Dataset


def base(dataset: Dataset, *, cv_n_splits, cv_groups_property=None,
         val_split_size=None, val_split_groups_property=None, n_repeats=1):

    def _apply_idx(x, subset):
        return x if subset is None or x is None else [x[i] for i in subset]

    def _get_groups(groups_property):
        if groups_property is not None:
            try:
                groups = getattr(dataset, groups_property)
            except AttributeError:
                raise('Dataset must have {} property containing groups\' ids'
                      ''.format(groups_property))
        else:
            groups = None
        return groups

    splits = []
    subj_ids = dataset.patient_ids
    groups = _get_groups(cv_groups_property)
    for i in range(n_repeats):
        kfoldClass = KFold if cv_groups_property is None else ShuffleGroupKFold
        kfold = kfoldClass(cv_n_splits, shuffle=True, random_state=i)
        for learn, test in kfold.split(X=subj_ids, groups=groups):
            if val_split_size is not None:
                splitClass = (ShuffleSplit if val_split_groups_property is None
                              else GroupShuffleSplit)
                split = splitClass(n_splits=1, test_size=val_split_size,
                                   random_state=i)
                subgroups = _get_groups(val_split_groups_property)
                train, val = next(split.split(
                    X=learn, groups=_apply_idx(subgroups, learn)))
                splits.append((_apply_idx(subj_ids, learn[train]),
                               _apply_idx(subj_ids, learn[val]),
                               _apply_idx(subj_ids, test)))
            else:
                splits.append((_apply_idx(subj_ids, learn),
                               _apply_idx(subj_ids, test)))
    return splits


class ShuffleGroupKFold(KFold):
    def split(self, *, X, groups):
        names_unique = np.unique(groups)

        for train, test in super().split(names_unique):
            train = np.in1d(groups, names_unique[train])
            test = np.in1d(groups, names_unique[test])
            yield np.where(train)[0], np.where(test)[0]


def get_cv_11(dataset: Dataset, *, n_splits):
    return base(dataset, cv_n_splits=n_splits)


def get_cv_111(dataset: Dataset, *, val_size, n_splits):
    return base(dataset, cv_n_splits=n_splits, val_split_size=val_size)


def get_group_cv_111(dataset: Dataset, *, val_part, n_splits):
    return base(dataset, cv_n_splits=n_splits, cv_groups_property='groups',
                val_split_size=val_part)
