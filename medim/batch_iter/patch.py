import numpy as np


def uniform(mscans, msegms, *, batch_size, patch_size_x, patch_size_y):
    """Patch iterator with uniformed distribution over spatial dimensions"""
    assert np.all(patch_size_x % 2 == patch_size_y % 2)
    patch_size_pad = (patch_size_x - patch_size_y) // 2

    n = len(mscans)

    max_spatial_idx = np.array([list(s.shape[1:])
                                for s in mscans]) - patch_size_y + 1

    x_batch = np.zeros((batch_size, mscans[0].shape[0], *patch_size_x),
                       dtype=np.float32)
    y_batch = np.zeros((batch_size, msegms[0].shape[0], *patch_size_y),
                       dtype=np.float32)

    while True:
        idx = np.random.randint(n, size=batch_size)
        start_idx = np.random.rand(batch_size, 3) * max_spatial_idx[idx]
        start_idx = np.int32(np.floor(start_idx))
        for i in range(batch_size):
            s = start_idx[i]
            slices = [...] + [slice(s[k], s[k] + patch_size_y[k])
                              for k in range(3)]
            y_batch[i] = msegms[idx[i]][slices]

            scan_shape = np.array(mscans[idx[i]].shape[1:])
            s = start_idx[i] - patch_size_pad
            e = s + patch_size_x
            padding_l = [0] + list(np.maximum(-s, 0))
            padding_r = [0] + list(np.maximum(e - scan_shape, 0))
            padding = tuple(zip(padding_l, padding_r))

            slices = [...] + [slice(max(s[k], 0), e[k]) for k in range(3)]
            min_const = np.min(mscans[idx[i]])
            x_batch[i] = np.pad(mscans[idx[i]][slices], padding,
                                mode='constant', constant_values=min_const)
        yield np.array(x_batch), np.array(y_batch)
