from typing import NamedTuple, Tuple, Union

import numpy as np
import torch

from experiments.data import INRDataset


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    aug_weights: Tuple
    aug_biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            aug_weights=tuple(w.to(device) for w in self.aug_weights),
            aug_biases=tuple(w.to(device) for w in self.aug_biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class SineINR2CoefDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.0,
        rotation_degree=90,
        noise_scale=2e-1,
        drop_rate=5e-1,
        resize_scale=0.0,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
    ):
        super().__init__(
            path=path,
            split=split,
            normalize=normalize,
            augmentation=augmentation,
            permutation=permutation,
            statistics_path=statistics_path,
            translation_scale=translation_scale,
            rotation_degree=rotation_degree,
            noise_scale=noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            class_mapping=class_mapping,
            quantile_dropout=quantile_dropout,
            pos_scale=pos_scale,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        path = self.dataset[item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = torch.from_numpy(np.array(state_dict["coef"]).astype(np.float32))

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        weights_aug, biases_aug = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])
        weights_aug = tuple([w.unsqueeze(-1) for w in weights_aug])
        biases_aug = tuple([b.unsqueeze(-1) for b in biases_aug])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)
            weights_aug, biases_aug = self._normalize(weights_aug, biases_aug)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(
            weights=weights,
            biases=biases,
            label=label,
            aug_weights=weights_aug,
            aug_biases=biases_aug,
        )
