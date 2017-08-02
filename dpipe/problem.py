from dpipe.datasets import Dataset


def make_loaders(dataset: Dataset, problem: str) -> (callable, callable, int):
    load_x = dataset.load_mscan
    if problem == 'segm':
        load_y, n_chans_out = dataset.load_segm, dataset.segm2msegm.shape[0]
    elif problem == 'msegm':
        load_y, n_chans_out = dataset.load_msegm, dataset.segm2msegm.shape[1]
    else:
        raise ValueError(
            f'Wrong problem: {problem}\nAvailable values are: "segm"; "msegm"'
        )
    return load_x, load_y, n_chans_out


def get_n_chans_out(dataset: Dataset, problem: str):
    if problem == 'segm':
        n_chans_out = dataset.segm2msegm.shape[0]
    elif problem == 'msegm':
        n_chans_out = dataset.segm2msegm.shape[1]
    else:
        raise ValueError(
            f'Wrong problem: {problem}\nAvailable values are: "segm"; "msegm"'
        )
    return n_chans_out
