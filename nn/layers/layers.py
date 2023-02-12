from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from nn.layers import (
    BiasToBiasBlock,
    BiasToWeightBlock,
    WeightToBiasBlock,
    WeightToWeightBlock,
)
from nn.layers.base import BaseLayer, GeneralSetLayer


class BN(nn.Module):
    def __init__(self, num_features, n_weights, n_biases):
        super().__init__()
        self.weights_bn = nn.ModuleList(
            nn.BatchNorm1d(num_features) for _ in range(n_weights)
        )
        self.biases_bn = nn.ModuleList(
            nn.BatchNorm1d(num_features) for _ in range(n_biases)
        )

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        new_weights, new_biases = [None] * len(weights), [None] * len(biases)
        for i, (m, w) in enumerate(zip(self.weights_bn, weights)):
            shapes = w.shape
            new_weights[i] = (
                m(w.permute(0, 3, 1, 2).flatten(start_dim=2))
                .permute(0, 2, 1)
                .reshape(shapes)
            )

        for i, (m, b) in enumerate(zip(self.biases_bn, biases)):
            new_biases[i] = m(b.permute(0, 2, 1)).permute(0, 2, 1)

        return tuple(new_weights), tuple(new_biases)


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        return tuple(F.relu(t) for t in weights), tuple(F.relu(t) for t in biases)


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        return tuple(F.leaky_relu(t) for t in weights), tuple(F.relu(t) for t in biases)


class Dropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        return tuple(F.dropout(t, p=self.p) for t in weights), tuple(
            F.dropout(t, p=self.p) for t in biases
        )


class DWSLayer(BaseLayer):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        in_features,
        out_features,
        bias=True,
        reduction="max",
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        add_skip=False,
        init_scale=1.0,
        init_off_diag_scale_penalty=1.0,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.weight_shapes = weight_shapes
        self.bias_shapes = bias_shapes
        self.n_matrices = len(weight_shapes) + len(bias_shapes)
        self.add_skip = add_skip

        self.weight_to_weight = WeightToWeightBlock(
            in_features,
            out_features,
            shapes=weight_shapes,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.bias_to_bias = BiasToBiasBlock(
            in_features,
            out_features,
            shapes=bias_shapes,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.bias_to_weight = BiasToWeightBlock(
            in_features,
            out_features,
            bias_shapes=bias_shapes,
            weight_shapes=weight_shapes,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )

        self.weight_to_bias = WeightToBiasBlock(
            in_features,
            out_features,
            bias_shapes=bias_shapes,
            weight_shapes=weight_shapes,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )

        self._init_model_params(init_scale, init_off_diag_scale_penalty)

        if self.add_skip:
            self.skip = self._get_mlp(in_features, out_features, bias=bias)
            with torch.no_grad():
                for m in self.skip.modules():
                    if isinstance(m, nn.Linear):
                        torch.nn.init.constant_(
                            m.weight, 1.0 / (m.weight.numel() ** 0.5)
                        )
                        torch.nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _apply_off_diag_penalty(name):
        if "weight_to_weight" in name or "bias_to_bias" in name:
            # for example mane='bias_to_bias.layers.0_0.layer.set_layer.mab.fc_q',
            # we extract the ["0", "0"] and check if the len of this set is of size 2
            # (here it is False, i.e., on the diag)
            return (len(set(name.split(".")[2].split("_"))) == 2) or (
                "skip" not in name
            )
        else:
            return True

    def _init_model_params(self, scale, off_diag_penalty=1.0):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                out_c, in_c = m.weight.shape
                g = (2 * in_c / out_c) ** 0.5
                # nn.init.xavier_normal_(m.weight, gain=g)
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                off_diag_penalty_ = (
                    off_diag_penalty if self._apply_off_diag_penalty(n) else 1.0
                )
                m.weight.data = m.weight.data * g * scale * off_diag_penalty_
                if m.bias is not None:
                    # m.bias.data.fill_(0.0)
                    m.bias.data.uniform_(-1e-4, 1e-4)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        new_weights_from_weights = self.weight_to_weight(weights)
        new_weights_from_biases = self.bias_to_weight(biases)

        new_biases_from_biases = self.bias_to_bias(biases)
        new_biases_from_weights = self.weight_to_bias(weights)

        # add and normalize by the number of matrices
        new_weights = tuple(
            (w0 + w1) / self.n_matrices
            for w0, w1 in zip(new_weights_from_weights, new_weights_from_biases)
        )
        new_biases = tuple(
            (b0 + b1) / self.n_matrices
            for b0, b1 in zip(new_biases_from_biases, new_biases_from_weights)
        )

        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            new_weights = tuple(ws + w for w, ws in zip(new_weights, skip_out[0]))
            new_biases = tuple(bs + b for b, bs in zip(new_biases, skip_out[1]))

        return new_weights, new_biases


class DownSampleDWSLayer(DWSLayer):
    def __init__(
        self,
        downsample_dim: int,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        in_features,
        out_features,
        bias=True,
        reduction="max",
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        add_skip=False,
        init_scale=1.0,
        init_off_diag_scale_penalty=1.0,
    ):
        d0 = weight_shapes[0][0]
        new_weight_shapes = list(weight_shapes)
        new_weight_shapes[0] = (downsample_dim, weight_shapes[0][1])

        super().__init__(
            weight_shapes=tuple(new_weight_shapes),
            bias_shapes=bias_shapes,
            in_features=in_features,
            out_features=out_features,
            reduction=reduction,
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
            add_skip=add_skip,
            init_scale=init_scale,
            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
        )

        self.downsample_dim = downsample_dim

        self.down_sample = GeneralSetLayer(
            in_features=d0,
            out_features=downsample_dim,
            reduction="attn",
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer="ds",
        )

        self.up_sample = GeneralSetLayer(
            in_features=downsample_dim,
            out_features=d0,
            reduction="attn",
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer="ds",
        )

        self.skip = self._get_mlp(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x

        # down-sample
        # (bs, d0, d1, in_features)
        w0 = weights[0]
        w0_skip = self.skip(w0)
        bs, d0, d1, _ = w0.shape
        # (bs, in_features, d1, d0)
        w0 = w0.permute(0, 3, 2, 1)
        # (bs, in_features, d1, downsample_dim)
        w0 = self.down_sample(w0)
        # (bs, downsample_dim, d1, in_features)
        w0 = w0.permute(0, 3, 2, 1)
        weights = list(weights)
        weights[0] = w0

        # cannibal layer out
        weights, biases = super().forward((tuple(weights), biases))

        # up-sample
        w0 = weights[0]
        # (bs, out_features, d1, downsample_dim)
        w0 = w0.permute(0, 3, 2, 1)
        # (bs, out_features, d1, d0)
        w0 = self.up_sample(w0)
        # (bs, d0, d1, out_features)
        w0 = w0.permute(0, 3, 2, 1)
        weights = list(weights)
        weights[0] = w0 + w0_skip  # add skip connection

        return weights, biases


class InvariantLayer(BaseLayer):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        in_features,
        out_features,
        bias=True,
        reduction="max",
        n_fc_layers=1,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.weight_shapes = weight_shapes
        self.bias_shapes = bias_shapes
        n_layers = len(weight_shapes) + len(bias_shapes)
        self.layer = self._get_mlp(
            in_features=(
                in_features * (n_layers - 3)
                +
                # in_features * d0 - first weight matrix
                in_features * weight_shapes[0][0]
                +
                # in_features * dL - last weight matrix
                in_features * weight_shapes[-1][-1]
                +
                # in_features * dL - last bias
                in_features * bias_shapes[-1][-1]
            ),
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        # first and last matrices are special
        first_w, last_w = weights[0], weights[-1]
        # first w is of shape (bs, d0, d1, in_features)
        # (bs, d1, d0 * in_features)
        pooled_first_w = first_w.permute(0, 2, 1, 3).flatten(start_dim=2)
        # (bs, d{L-1}, dL * in_features)
        pooled_last_w = last_w.flatten(start_dim=2)
        # (bs, d0 * in_features)
        pooled_first_w = self._reduction(pooled_first_w, dim=1)
        # (bs, dL * in_features)
        pooled_last_w = self._reduction(pooled_last_w, dim=1)
        # last bias is special
        last_b = biases[-1]
        # (bs, dL * in_features)
        pooled_last_b = last_b.flatten(start_dim=1)

        # concat
        pooled_weights = torch.cat(
            [
                self._reduction(w.permute(0, 3, 1, 2).flatten(start_dim=2), dim=2)
                for w in weights[1:-1]
            ],
            dim=-1,
        )  # (bs, (len(weights) - 2) * in_features)
        # (bs, (len(weights) - 2) * in_features + d0 * in_features + dL * in_features)
        pooled_weights = torch.cat(
            (pooled_weights, pooled_first_w, pooled_last_w), dim=-1
        )

        pooled_biases = torch.cat(
            [self._reduction(b, dim=1) for b in biases[:-1]], dim=-1
        )  # (bs, (len(biases) - 1) * in_features)
        # (bs, (len(biases) - 1) * in_features + dL * in_features)
        pooled_biases = torch.cat((pooled_biases, pooled_last_b), dim=-1)

        pooled_all = torch.cat(
            [pooled_weights, pooled_biases], dim=-1
        )  # (bs, (num layers - 3) * in_features + d0 * in_features + dL * in_features + dL * in_features)
        return self.layer(pooled_all)


class NaiveInvariantLayer(BaseLayer):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        in_features,
        out_features,
        bias=True,
        reduction="max",
        n_fc_layers=1,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.weight_shapes = weight_shapes
        self.bias_shapes = bias_shapes
        n_layers = len(weight_shapes) + len(bias_shapes)
        self.layer = self._get_mlp(
            in_features=in_features * n_layers, out_features=out_features, bias=bias
        )

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weights, biases = x
        pooled_weights = torch.cat(
            [
                self._reduction(w.permute(0, 3, 1, 2).flatten(start_dim=2), dim=2)
                for w in weights
            ],
            dim=-1,
        )  # (bs, len(weights) * in_features)
        pooled_biases = torch.cat(
            [self._reduction(b, dim=1) for b in biases], dim=-1
        )  # (bs, len(biases) * in_features)
        pooled_all = torch.cat(
            [pooled_weights, pooled_biases], dim=-1
        )  # (bs, num layers * in_features)
        return self.layer(pooled_all)
