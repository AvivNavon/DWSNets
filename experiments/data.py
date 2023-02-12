import json
import random
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from experiments.utils import make_coordinates
from nn.inr import INR


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
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
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class ImageBatch(NamedTuple):
    image: torch.Tensor
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.image)


class INRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-1,
        drop_rate=1e-2,
        resize_scale=0.2,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
    ):
        # assert split in ["test", "train"]
        self.split = split
        self.dataset = json.load(open(path, "r"))[self.split]

        self.augmentation = augmentation
        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.translation_scale = translation_scale
        self.rotation_degree = rotation_degree
        self.noise_scale = noise_scale
        self.drop_rate = drop_rate
        self.resize_scale = resize_scale
        self.pos_scale = pos_scale
        self.quantile_dropout = quantile_dropout

        if class_mapping is not None:
            self.class_mapping = class_mapping
            self.dataset["label"] = [
                self.class_mapping[l] for l in self.dataset["label"]
            ]

    def __len__(self):
        return len(self.dataset["label"])

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # translation
        translation = torch.empty(weights[0].shape[0]).uniform_(
            -self.translation_scale, self.translation_scale
        )
        order = random.sample(range(1, len(weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = bias_res @ weights[i]

        new_biases[i] += bias_res

        # rotation
        if new_weights[0].shape[0] == 2:
            rot_mat = self.rotation_mat(self.rotation_degree)
            new_weights[0] = rot_mat @ new_weights[0]

        # noise
        new_weights = [w + w.std() * self.noise_scale for w in new_weights]
        new_biases = [
            b + b.std() * self.noise_scale if b.shape[0] > 1 else b for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # scale
        # todo: can also apply to deeper layers
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        new_weights[0] = new_weights[0] * rand_scale

        # positive scale
        if self.pos_scale > 0:
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = new_weights[i] * scale
                new_biases[i] = new_biases[i] * scale
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T

        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return new_weights, new_biases

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        label = int(self.dataset["label"][item])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)


class INRStateDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-1,
        drop_rate=1e-2,
        resize_scale=0.2,
        pos_scale=0.0,
        class_mapping=None,
    ):
        super().__init__(
            path,
            split,
            normalize,
            augmentation,
            permutation,
            statistics_path,
            translation_scale,
            rotation_degree,
            noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            class_mapping=class_mapping,
            pos_scale=pos_scale,
        )

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = int(self.dataset["label"][item])
        layer_names, layer_params = list(state_dict.keys()), list(state_dict.values())
        return layer_names, layer_params, label, path


class INRImageDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        drop_rate=1e-2,
        noise_scale=1e-1,
        resize_scale=0.2,
        pos_scale=0.0,
        inr_class=INR,
        inr_kwargs={"n_layers": 3, "in_dim": 2, "up_scale": 16},
        image_size=(28, 28),
        class_mapping=None,
    ):
        super().__init__(
            path,
            split,
            normalize,
            augmentation,
            permutation,
            statistics_path,
            translation_scale,
            rotation_degree,
            noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            class_mapping=class_mapping,
            pos_scale=pos_scale,
        )
        self.inr = inr_class(**inr_kwargs)
        self.image_size = image_size

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location="cpu")

        weight_names = [k for k in state_dict.keys() if "weight" in k]
        bias_names = [k for k in state_dict.keys() if "bias" in k]

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        label = int(self.dataset["label"][item])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        new_state_dict = {}
        for i, k in enumerate(weight_names):
            new_state_dict[k] = weights[i].squeeze(-1).permute(1, 0)

        for i, k in enumerate(bias_names):
            new_state_dict[k] = biases[i].squeeze(-1)

        self.inr.load_state_dict(new_state_dict)
        self.inr.eval()
        input = make_coordinates(self.image_size, 1)
        with torch.no_grad():
            image = self.inr(input)
            image = image.view(*self.image_size, -1)
            image = image.permute(2, 0, 1)
        return ImageBatch(image=image, label=label)
