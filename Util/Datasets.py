import numpy as np


class Dataset:
    def __init__(self, train, test, gt_labels=None, name=None):
        self.train = train
        self.test = test
        self.gt_labels = gt_labels
        self.name = name

        if gt_labels is not None:
            if train is not None:
                if np.max(train[-1]) >= len(gt_labels):
                    raise ValueError(
                        "Had {} labels, but maximum class label was {}".format(len(gt_labels), np.max(train[-1])))
            if test is not None:
                if np.max(test[-1]) >= len(gt_labels):
                    raise ValueError

    def normalized_linear(self, epsilon=1e-15):
        import numpy as np

        new_train = None
        new_test = None

        def norm(d):
            assert len(d.shape) == 2
            if not np.issubdtype(d.dtype, np.floating):
                raise ValueError("Invalid dtype: {}".format(d.dtype))
            min = np.min(d, axis=0)
            max = np.max(d, axis=0)
            return (d - min) / np.maximum(max - min, epsilon)

        if self.train is not None:
            if len(self.train) != 2:
                raise ValueError("Unsupported train format, expected (x, y)-tuple")
            new_train = norm(self.train[0]), self.train[1]

        if self.test is not None:
            if len(self.test) != 2:
                raise ValueError("Unsupported test format, expected (x, y)-tuple")
            new_test = norm(self.test[0]), self.test[1]

        return Dataset(new_train, new_test, gt_labels=self.gt_labels, name=self.name)

    def transform(self, fn, gt_labels=None, name=None):
        train_new = fn(self.train) if self.train is not None else None
        test_new = fn(self.test) if self.test is not None else None

        gt_labels_new = gt_labels if gt_labels is not None else self.gt_labels
        name_new = name if name is not None else self.name

        return Dataset(train_new, test_new, gt_labels=gt_labels_new, name=name_new)

    def filtered(self, labels=None):
        import numpy as np

        if labels is None:
            raise ValueError("Must specify labels")

        if not all((l in self.gt_labels for l in labels)):
            raise ValueError
        ids = np.array([self.gt_labels.index(l) for l in labels])

        reverse_ids = np.zeros(np.max(ids) + 1, dtype=int)

        for i, v in enumerate(ids):
            reverse_ids[v] = i

        def func(dataset):
            x, y = dataset

            if len(y.shape) != 1:
                raise ValueError("Unsupported ground truth shape: {}".format(y.shape))

            mask = np.any(y[:, None] == ids, axis=-1)
            return x[mask], reverse_ids[y[mask]]

        return self.transform(func, gt_labels=labels)


def proportional_sample_preserve_all(x, y, n_total):
    weights = np.ones(len(y))

    idxs = []
    uniqs = np.unique(y)

    for u in uniqs:
        i = np.random.choice(np.nonzero(y == u)[0])
        idxs.append(i)
        weights[i] = 0

    weights = weights / np.sum(weights)
    idxs = np.concatenate([np.array(idxs), np.random.choice(np.arange(len(y)), n_total - len(idxs), replace=False, p=weights)])

    assert len(idxs) == n_total
    return x[idxs], y[idxs]


def load_text_data(path, name, shuffle=True, max_total=2000):
    total = np.loadtxt(path)

    if shuffle:
        np.random.shuffle(total)

    # Last column should be label, subtract minimum to always have 0 indexing
    total[:, -1] = total[:, -1] - np.min(total[:, -1])
    x, y = total[..., :-1], total[..., -1].astype(int)

    if 0 < max_total < len(y):
        x, y = proportional_sample_preserve_all(x, y, max_total)

    labels = [str(it) for it in np.unique(y)]
    return Dataset((x, y), None, gt_labels=labels, name=name)


def load_moons(n_total=2000):
    from sklearn.datasets import make_moons

    x, y = make_moons(n_samples=n_total, noise=.05)
    return Dataset((x, y), None, gt_labels=['1', '2'], name='moons')


def load_pa(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    result = []
    is_start = False

    for l in lines:
        if is_start:
            result.append(int(l))
        elif l.startswith('---'):
            is_start = True

    return np.array(result) - np.min(result)


def load_s4(n_total=1000):
    import os

    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    x = np.loadtxt(os.path.join(data_folder, 's4.txt'))
    y = load_pa(os.path.join(data_folder, 's4-label.pa'))
    assert len(x) == len(y)

    if n_total is not None:
        idxs = np.random.choice(np.arange(len(x)), n_total)
        x, y = x[idxs], y[idxs]

    return Dataset((x, y), None, gt_labels=[str(i) for i in np.unique(y)], name='s4')


def load_embeddings(which='mnist', sample=None):
    import h5py
    import os

    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dkm-{}.hdf5'.format(which))

    try:
        with h5py.File(path, "r") as h5f:
            x, y = np.array(h5f['embeddings']), np.array(h5f['y_true'])
            labels = [s for s in h5f['labels']]
    except Exception as e:
        raise IOError("Failed to load {}".format(which)) from e

    if sample == 'default':
        idxs = np.random.choice(np.arange(len(y)), 2000, replace=False)
        x, y = x[idxs], y[idxs]
    elif sample is None:
        pass
    else:
        raise ValueError

    return Dataset((x, y), None, gt_labels=labels, name=which)
