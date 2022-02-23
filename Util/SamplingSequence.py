import numpy as np
from tensorflow.keras.utils import Sequence


class SamplingSequence(Sequence):
    def __init__(self, batch_size, x, y):
        super(SamplingSequence, self).__init__()

        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.idxs = np.array([i for i in range(len(x))])
        self.length = len(x) // self.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        indices = np.random.choice(self.idxs, size=self.batch_size)
        return self.x[indices], self.y[indices]
