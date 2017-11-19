import numpy as np

from .cv import ShuffleGroupKFold, train_test_split_groups
from sklearn.model_selection import KFold
from dpipe.dataset import DataSet
from dpipe.config import register


@register()
def get_subj_ids(dataset: DataSet):
    """Returns a list of subject IDs """
    return dataset.ids


@register()
def get_groups(dataset: DataSet, groups_property='groups'):
    """Returns a list which will be used to perform a group-based CV split"""
    try:
        groups = getattr(dataset, groups_property)
    except AttributeError:
        raise ('Dataset must have {} property containing groups\' ids'
               ''.format(groups_property))
    return groups


@register()
def kfold_split(subj_ids, n_splits, groups=None, **kwargs):
    """Perform kfold (or group kfold) split
    Returns indices: [[train_0, val_0], ..., [train_k, val_k]]
    """
    kfoldClass = KFold if groups is None else ShuffleGroupKFold
    kfold = kfoldClass(n_splits, **kwargs)
    return [split for split in kfold.split(X=subj_ids, groups=groups)]


@register()
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


@register()
def indices_to_subj_ids(splits, subj_ids):
    """Converts split indices to subject IDs"""
    return [list(map(lambda ids: [subj_ids[i] for i in ids], split))
            for split in splits]


@register()
def get_loo_cv(dataset: DataSet, *, val_size=None):
    """Leave one group out CV. Validation subset will be selected randomly"""
    subj_ids = get_subj_ids(dataset)
    groups = get_groups(dataset)
    n_splits = len(np.unique(groups))
    splits = kfold_split(subj_ids, n_splits, groups=groups)
    if val_size is not None:
        splits = split_train(splits, val_size)
    print(indices_to_subj_ids(splits, subj_ids))
    return indices_to_subj_ids(splits, subj_ids)


# non registered examples
def get_cv_11(dataset: DataSet, *, n_splits):
    subj_ids = get_subj_ids(dataset)
    split_indices = kfold_split(subj_ids, n_splits)
    return indices_to_subj_ids(split_indices, subj_ids)


def get_cv_111(dataset: DataSet, *, val_size, n_splits):
    subj_ids = get_subj_ids(dataset=dataset)
    split_indices = kfold_split(subj_ids=subj_ids, n_splits=n_splits)
    split_indices = split_train(splits=split_indices, val_size=val_size)
    return indices_to_subj_ids(splits=split_indices, subj_ids=subj_ids)


def get_group_cv_111(dataset: DataSet, *, val_size, n_splits):
    subj_ids = get_subj_ids(dataset)
    groups = get_groups(dataset)
    split_indices = kfold_split(subj_ids, n_splits, groups=groups)
    split_indices = split_train(split_indices, val_size, groups=groups)
    return indices_to_subj_ids(split_indices, subj_ids)
