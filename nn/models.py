from typing import Tuple

import torch
from torch import nn

from nn.layers import BN, DownSampleDWSLayer, Dropout, DWSLayer, InvariantLayer, ReLU


class MLPModel(nn.Module):
    def __init__(self, in_dim=2208, hidden_dim=256, n_hidden=2, bn=False, init_scale=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for i in range(n_hidden):
            if i < n_hidden - 1:
                if not bn:
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                else:
                    layers.extend(
                        [
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(),
                        ]
                    )
            else:
                layers.append(nn.Linear(hidden_dim, in_dim))
        # # todo: this model have one extra layer compare with the other alternatives for model-to-model
        # layers.append(nn.Linear(hidden_dim, in_dim))
        self.seq = nn.Sequential(*layers)

        self._init_model_params(init_scale)

    def _init_model_params(self, scale):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                out_c, in_c = m.weight.shape
                g = (2 * in_c / out_c) ** 0.5
                # nn.init.xavier_normal_(m.weight, gain=g)
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                m.weight.data = m.weight.data * g * scale
                if m.bias is not None:
                    # m.bias.data.fill_(0.0)
                    m.bias.data.uniform_(-1e-4, 1e-4)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        bs = weight[0].shape[0]
        weight_shape, bias_shape = [w[0, :].shape for w in weight], [
            b[0, :].shape for b in bias
        ]
        all_weights = weight + bias
        weight = torch.cat([w.flatten(start_dim=1) for w in all_weights], dim=-1)
        weights_and_biases = self.seq(weight)
        n_weights = sum([w.numel() for w in weight_shape])
        weights = weights_and_biases[:, :n_weights]
        biases = weights_and_biases[:, n_weights:]
        weight, bias = [], []
        w_index = 0
        for s in weight_shape:
            weight.append(weights[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        w_index = 0
        for s in bias_shape:
            bias.append(biases[:, w_index : w_index + s.numel()].reshape(bs, *s))
            w_index += s.numel()
        return tuple(weight), tuple(bias)


class MLPModelForClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_hidden=2, n_classes=10, bn=False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden):
            if not bn:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    ]
                )

        layers.append(nn.Linear(hidden_dim, n_classes))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        weight, bias = x
        all_weights = weight + bias
        weight = torch.cat([w.flatten(start_dim=1) for w in all_weights], dim=-1)
        return self.seq(weight)


class DWSModel(nn.Module):
    def __init__(
            self,
            weight_shapes: Tuple[Tuple[int, int], ...],
            bias_shapes: Tuple[
                Tuple[
                    int,
                ],
                ...,
            ],
            input_features,
            hidden_dim,
            n_hidden=2,
            output_features=None,
            reduction="max",
            bias=True,
            n_fc_layers=1,
            num_heads=8,
            set_layer="sab",
            input_dim_downsample=None,
            dropout_rate=0.0,
            add_skip=False,
            add_layer_skip=False,
            init_scale=1e-4,
            init_off_diag_scale_penalty=1.,
            bn=False,
            diagonal=False,
    ):
        super().__init__()
        assert len(weight_shapes) > 2, "the current implementation only support input networks with M>2 layers."

        self.input_features = input_features
        self.input_dim_downsample = input_dim_downsample
        if output_features is None:
            output_features = hidden_dim

        self.add_skip = add_skip
        if self.add_skip:
            self.skip = nn.Linear(
                input_features,
                output_features,
                bias=bias
            )
            with torch.no_grad():
                torch.nn.init.constant_(self.skip.weight, 1. / self.skip.weight.numel())
                torch.nn.init.constant_(self.skip.bias, 0.)

        if input_dim_downsample is None:
            layers = [
                DWSLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [

                        ReLU(),
                        Dropout(dropout_rate),
                        DWSLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim if i != (n_hidden - 1) else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        else:
            layers = [
                DownSampleDWSLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    downsample_dim=input_dim_downsample,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        DownSampleDWSLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim if i != (n_hidden - 1) else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            downsample_dim=input_dim_downsample,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        out = self.layers(x)
        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            weight_out = tuple(ws + w for w, ws in zip(out[0], skip_out[0]))
            bias_out = tuple(bs + b for b, bs in zip(out[1], skip_out[1]))
            out = weight_out, bias_out
        return out


class DWSModelForClassification(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[
                int,
            ],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        n_classes=10,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        n_out_fc=1,
        dropout_rate=0.0,
        input_dim_downsample=None,
        init_scale=1.,
        init_off_diag_scale_penalty=1.,
        bn=False,
        add_skip=False,
        add_layer_skip=False,
        equiv_out_features=None,
        diagonal=False,
    ):
        super().__init__()
        self.layers = DWSModel(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            reduction=reduction,
            bias=bias,
            output_features=equiv_out_features,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
            dropout_rate=dropout_rate,
            input_dim_downsample=input_dim_downsample,
            init_scale=init_scale,
            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            bn=bn,
            add_skip=add_skip,
            add_layer_skip=add_layer_skip,
            diagonal=diagonal,
        )
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()
        self.clf = InvariantLayer(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            in_features=hidden_dim
            if equiv_out_features is None
            else equiv_out_features,
            out_features=n_classes,
            reduction=reduction,
            n_fc_layers=n_out_fc,
        )

    def forward(
        self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]], return_equiv=False
    ):
        x = self.layers(x)
        out = self.clf(self.dropout(self.relu(x)))
        if return_equiv:
            return out, x
        else:
            return out


if __name__ == "__main__":
    weights = (
        torch.randn(4, 784, 128, 1),
        torch.randn(4, 128, 128, 1),
        torch.randn(4, 128, 10, 1),
    )
    biases = (torch.randn(4, 128, 1), torch.randn(4, 128, 1), torch.randn(4, 10, 1))
    in_dim = sum([w[0, :].numel() for w in weights]) + sum(
        [w[0, :].numel() for w in biases]
    )
    weight_shapes = tuple(w.shape[1:3] for w in weights)
    bias_shapes = tuple(b.shape[1:2] for b in biases)
    n_params = sum([i.numel() for i in weight_shapes + bias_shapes])
