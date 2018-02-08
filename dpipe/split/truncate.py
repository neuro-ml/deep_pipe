def truncate(to_truncate, experiments_num):
    """
    A wrapper for splitter that reduces the number of folds by leaving only experiments_num number of folds.
    """
    return to_truncate[:experiments_num]
