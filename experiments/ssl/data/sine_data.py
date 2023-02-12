from typing import NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset


class SineBatch(NamedTuple):
    x: Union[float, torch.Tensor]
    y: Union[float, torch.Tensor]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.x)


class SineDataset(Dataset):
    def __init__(self, coef, n_samples=1000):
        self.n_samples = n_samples
        self.coef = coef

        self.x = np.arange(-np.pi, np.pi, np.pi / self.n_samples)
        self.y = self.coef[0] * np.sin(
            self.x * 2 * np.pi / (1 / self.coef[1] * 2 * np.pi)
        )

    def __getitem__(self, item):
        return SineBatch(
            torch.tensor([self.x[item]], dtype=torch.float32),
            torch.tensor([self.y[item]], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.x)

    def plot(self):
        plt.scatter(self.x, self.y, s=0.5)
        plt.show()


if __name__ == "__main__":
    data = SineDataset(coef=[5, 10])
    data.plot()
