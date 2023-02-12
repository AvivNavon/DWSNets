from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from nn.layers.base import BaseLayer, GeneralSetLayer


class SameLayer(BaseLayer):
    """Mapping Wi -> bi"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
        is_input_layer=False,
        is_output_layer=False,
    ):
        """

        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        :param num_heads:
        :param set_layer:
        :param is_input_layer: indicates that the weight (and bias) is that of the last layer.
        :param is_output_layer: indicates that the weight (and bias) is that of the last layer.
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.is_output_layer = is_output_layer
        self.is_input_layer = is_input_layer
        assert not (is_input_layer and is_output_layer)

        if self.is_input_layer:
            self.layer = GeneralSetLayer(
                in_features=in_features * in_shape[0],  # d0 * in_features
                out_features=out_features,  # out_features
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                num_heads=num_heads,
                set_layer=set_layer,
            )
        elif self.is_output_layer:
            self.layer = self._get_mlp(
                in_features=in_features * out_shape[-1],  # dL * in_features
                out_features=out_features * out_shape[-1],  # dL * out_features
                bias=bias,
            )
        else:
            # i != 0, L-1
            self.layer = GeneralSetLayer(
                in_features=in_features,
                out_features=out_features,
                reduction=reduction,
                bias=bias,
                n_fc_layers=n_fc_layers,
                num_heads=num_heads,
                set_layer=set_layer,
            )

    def forward(self, x):
        if self.is_input_layer:
            # (bs, d0, d1, in_features)
            # (bs, d1, d0 * in_features)
            x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
            # (bs, d1, out_features)
            x = self.layer(x)

        elif self.is_output_layer:
            # (bs, d{L-1}, dL, in_features)
            # (bs, dL, in_features)
            x = self._reduction(x, dim=1)
            # (bs, dL * in_features)
            x = x.flatten(start_dim=1)
            # (bs, dL * out_features)
            x = self.layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)

        else:
            # (bs, di, d{i+1}, in_features)
            # (bs, d{i+1}, in_features)
            x = self._reduction(x, dim=1)
            # (bs, d{i+1}, out_features)
            x = self.layer(x)

        return x


class SuccessiveLayers(BaseLayer):
    """Mapping Wi -> bj where i=j+1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
        first_dim_is_output=False,
    ):
        """

        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        :param num_heads:
        :param set_layer:
        :param first_dim_is_output: first dim (i) is the output layer for the INR (i=L-1)
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.first_dim_is_output = first_dim_is_output
        if self.first_dim_is_output:
            # i=L-1, j=L-2
            in_features = self.in_features * in_shape[-1]  # in_features * dL
            out_features = self.out_features
        else:
            # i != L-1
            in_features = self.in_features
            out_features = self.out_features

        self.layer = GeneralSetLayer(
            in_features=in_features,
            out_features=out_features,
            reduction=reduction,
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )

    def forward(self, x):
        if self.first_dim_is_output:
            # (bs, d{L-1}, dL, in_features)
            # (bs, d{L-1}, dL * in_features)
            x = x.flatten(start_dim=2)
            # (bs, d{L-1}, out_features)
            x = self.layer(x)

        else:
            # (bs, di, d{i+1}, in_features)
            # (bs, di, in_features)
            x = self._reduction(x, dim=2)
            # (bs, di, out_features)
            x = self.layer(x)

        return x


class NonNeighborInternalLayer(BaseLayer):
    """Mapping Wi -> bj where i != j, j+1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        first_dim_is_input=False,
        first_dim_is_output=False,
        last_dim_is_output=False,
    ):
        """

        :param in_features: input feature dim
        :param out_features:
        :param in_shape:
        :param out_shape:
        :param bias:
        :param reduction:
        :param n_fc_layers:
        :param first_dim_is_input: indicates that the input matrix is the weight matrix for the first layer.
        :param first_dim_is_output: indicates that the output matrix is the weight matrix for the last layer.
        :param last_dim_is_output: indicates that the output matrix is the weight matrix for the last layer.
        """
        super().__init__(
            in_features,
            out_features,
            in_shape=in_shape,
            out_shape=out_shape,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.first_dim_is_input = first_dim_is_input
        self.first_dim_is_output = first_dim_is_output
        self.last_dim_is_output = last_dim_is_output

        if self.first_dim_is_input:
            # i = 0
            if self.last_dim_is_output:
                # i = 0, j = L - 1
                in_features = self.in_features * in_shape[0]  # in_features * d0
                out_features = self.out_features * out_shape[0]  # out_features * dL
            else:
                # i = 0, j != L-1
                in_features = self.in_features * in_shape[0]  # in_features * d0
                out_features = self.out_features  # out_features

        elif self.first_dim_is_output:
            # i = L-1
            in_features = self.in_features * in_shape[-1]  # in_features * dL
            out_features = self.out_features  # out_features

        elif self.last_dim_is_output:
            # j = L-1, i != 0
            in_features = self.in_features  # in_features
            out_features = self.out_features * self.out_shape[-1]  # out_features * dL

        else:
            # i != L-1, 0 and  j != L-1
            in_features = self.in_features  # in_features
            out_features = self.out_features  # out_features

        self.layer = self._get_mlp(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x):
        if self.first_dim_is_input:
            # i = 0
            if self.last_dim_is_output:
                # i = 0, j = L-1
                # (bs, d0, d1, in_features)
                # (bs, d0, in_features)
                x = self._reduction(x, dim=2)
                # (bs, d0 * in_features)
                x = x.flatten(start_dim=1)
                # (bs, dL * out_features)
                x = self.layer(x)
                # (bs, dL, out_features)
                x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
            else:
                # i = 0, j != L-1
                # (bs, d0, d1, in_features)
                # (bs, d0, in_features)
                x = self._reduction(x, dim=2)
                # (bs, d0 * in_features)
                x = x.flatten(start_dim=1)
                # (bs, out_features)
                x = self.layer(x)
                # (bs, d{j+1}, out_features)
                x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1)

        elif self.first_dim_is_output:
            # i = L-1
            # (bs, d{L-1}, dL, in_features)
            # (bs, dL, in_features)
            x = self._reduction(x, dim=1)
            # (bs, dL * in_features)
            x = x.flatten(start_dim=1)
            # (bs, out_features)
            x = self.layer(x)
            # (bs, d{j+1}, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1)

        elif self.last_dim_is_output:
            # j = L-1, i != 0
            # (bs, di, d{i+1}, in_features)
            # (bs, in_features, di * d{i+1})
            x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
            # (bs, in_features)
            x = self._reduction(x, dim=2)
            # (bs, dL * out_features)
            x = self.layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)

        else:
            # i != L-1, 0 and  j != L-1
            # (bs, di, d{i+1}, in_features)
            # (bs, in_features, di * d{i+1})
            x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
            # (bs, in_features)
            x = self._reduction(x, dim=2)
            # (bs, out_features)
            x = self.layer(x)
            # (bs, d{j+1}, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1)

        return x


class WeightToBiasBlock(BaseLayer):
    """Wi -> bj"""

    def __init__(
        self,
        in_features,
        out_features,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )
        assert all([len(shape) == 1 for shape in bias_shapes])
        assert all([len(shape) == 2 for shape in weight_shapes])
        assert len(bias_shapes) == len(weight_shapes)

        self.weight_shapes = weight_shapes
        self.bias_shapes = bias_shapes
        self.n_layers = len(bias_shapes)

        self.layers = ModuleDict()
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if i == j:
                    self.layers[f"{i}_{j}"] = SameLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=weight_shapes[i],
                        out_shape=bias_shapes[j],
                        reduction=reduction,
                        bias=bias,
                        num_heads=num_heads,
                        set_layer=set_layer,
                        n_fc_layers=n_fc_layers,
                        is_input_layer=(
                            i == 0
                        ),  # todo: make sure this condition is correct
                        is_output_layer=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif i == j + 1:
                    self.layers[f"{i}_{j}"] = SuccessiveLayers(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=weight_shapes[i],
                        out_shape=bias_shapes[j],
                        reduction=reduction,
                        bias=bias,
                        num_heads=num_heads,
                        set_layer=set_layer,
                        n_fc_layers=n_fc_layers,
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                else:
                    self.layers[f"{i}_{j}"] = NonNeighborInternalLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=weight_shapes[i],
                        out_shape=bias_shapes[j],
                        reduction=reduction,
                        bias=bias,
                        # todo: make sure this condition is correct
                        first_dim_is_input=(i == 0),
                        first_dim_is_output=(i == self.n_layers - 1),
                        last_dim_is_output=(j == self.n_layers - 1),
                    )

    def forward(self, x: Tuple[torch.tensor]):
        out_weights = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                out_weights[j] = out_weights[j] + self.layers[f"{i}_{j}"](x[i])

        return tuple(out_weights)
