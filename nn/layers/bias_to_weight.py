from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from nn.layers.base import BaseLayer, GeneralSetLayer


class SameLayer(BaseLayer):
    """Mapping bi -> Wi"""

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
        :param is_output_layer: indicates that the bias is that of the last layer.
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
                in_features=in_features,
                out_features=out_features * out_shape[0],  # d0 * out_features
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
            # (bs, d1, in_features)
            # (bs, d1, d0 * out_features)
            x = self.layer(x)
            # (bs, d1, d0, out_features)
            x = x.reshape(
                x.shape[0], self.out_shape[-1], self.out_shape[0], self.out_features
            )
            # (bs, d0, d1, out_features)
            x = x.permute(0, 2, 1, 3)

        elif self.is_output_layer:
            # (bs, dL, in_features)
            # (bs, dL * in_features)
            x = x.flatten(start_dim=1)
            # (bs, dL * out_features)
            x = self.layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
            # (bs, d{L-1}, dL, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1, 1)
        else:
            # (bs, di, in_features)
            # (bs, di, out_features)
            x = self.layer(x)
            # (bs, d{i-1}, di, out_features)
            x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1, 1)
        return x


class SuccessiveLayers(BaseLayer):
    """Mapping bi -> Wj where i=j-1"""

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
        :param num_heads:
        :param set_layer:
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
        self.last_dim_is_output = last_dim_is_output
        if self.last_dim_is_output:
            in_features = self.in_features
            out_features = self.out_features * out_shape[1]  # dL * out_features
            # j=L-1, i=L-2, bi is of shape d{L-1}, Wj is of shape (d{L-1}, dL)
        else:
            # j!=L-1, bi is of shape d{i+1}, Wj is of shape (d{i+1}, d{i+2})
            in_features = self.in_features
            out_features = self.out_features  # out_features

        self.layer = GeneralSetLayer(
            in_features=in_features,
            out_features=out_features,  # dL * out_features
            reduction=reduction,
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )

    def forward(self, x):
        if self.last_dim_is_output:
            # (bs, d{L-1}, in_features)
            # (bs, d{L-1}, dL * out_features)
            x = self.layer(x)
            # (bs, d{L-1}, dL * out_features)
            x = x.reshape(x.shape[0], *self.out_shape, self.out_features)
        else:
            # (bs, d{i+1}, in_features)
            # (bs, d{i+1}, out_features)
            x = self.layer(x)
            # (bs, d{i+1}, d{i+2} out_features)
            x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)
        return x


class NonNeighborInternalLayer(BaseLayer):
    """Mapping bi -> Wj where i != j,j-1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        last_dim_is_input=False,
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
        # todo: add assertions to varify boolean conditions are OK
        self.last_dim_is_input = last_dim_is_input
        self.first_dim_is_output = first_dim_is_output

        if self.first_dim_is_output:
            # i = L-1
            if self.last_dim_is_input:
                # i = L-1, j = 0
                in_features = self.in_features * in_shape[-1]  # in_features * dL
                out_features = self.out_features * out_shape[0]  # out_features * d0

            else:
                # i = L-1, j != 0
                in_features = self.in_features * in_shape[-1]  # in_features * dL
                out_features = self.out_features  # out_features

        else:
            # i != L-1
            if self.last_dim_is_input:
                # i != L-1, j = 0
                in_features = self.in_features  # in_features
                out_features = self.out_features * out_shape[0]  # out_features * d0

            else:
                # i != L-1, j != 0
                in_features = self.in_features  # in_features
                out_features = self.out_features  # out_features

        self.layer = self._get_mlp(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x):
        if self.first_dim_is_output:
            # i = L-1
            if self.last_dim_is_input:
                # i = L-1, j = 0
                # (bs, dL, in_features)
                # (bs, dL * in_features)
                x = x.flatten(start_dim=1)
                # (bs, d0 * out_features)
                x = self.layer(x)
                # (bs, d0, out_features)
                x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
                # (bs, d0, d1, out_features)
                x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)
            else:
                # i = L-1, j = 0
                # (bs, dL, in_features)
                # (bs, dL * in_features)
                x = x.flatten(start_dim=1)
                # (bs, out_features)
                x = self.layer(x)
                # (bs, dj, d{j+1}, out_features)
                x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
        else:
            if self.last_dim_is_input:
                # i != L-1, j = 0
                # (bs, d{i+1}, in_features)
                # (bs, in_features)
                x = self._reduction(x, dim=1)
                # (bs, d0 * out_shape)
                x = self.layer(x)
                # (bs, d0, out_shape)
                x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
                # (bs, d0, d1, out_features)
                x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)
            else:
                # i != L-1, j != 0
                # (bs, d{i+1}, in_features)
                # (bs, in_features)
                x = self._reduction(x, dim=1)
                # (bs, out_shape)
                x = self.layer(x)
                # (bs, dj, d{j+1}, out_features)
                x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
        return x


class BiasToWeightBlock(BaseLayer):
    """bi -> Wj"""

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
        diagonal: bool = False,
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
        self.diagonal = diagonal

        self.layers = ModuleDict()
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and not ((i == j) or (i == j - 1)):
                    continue
                if i == j:
                    self.layers[f"{i}_{j}"] = SameLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=bias_shapes[i],
                        out_shape=weight_shapes[j],
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
                elif i == j - 1:
                    self.layers[f"{i}_{j}"] = SuccessiveLayers(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=bias_shapes[i],
                        out_shape=weight_shapes[j],
                        reduction=reduction,
                        bias=bias,
                        num_heads=num_heads,
                        set_layer=set_layer,
                        n_fc_layers=n_fc_layers,
                        last_dim_is_output=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                else:
                    self.layers[f"{i}_{j}"] = NonNeighborInternalLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=bias_shapes[i],
                        out_shape=weight_shapes[j],
                        reduction=reduction,
                        bias=bias,
                        last_dim_is_input=(
                            j == 0
                        ),  # todo: make sure this condition is correct
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )

    def forward(self, x: Tuple[torch.tensor]):
        out_weights = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and not ((i == j) or (i == j - 1)):
                    continue
                out_weights[j] = out_weights[j] + self.layers[f"{i}_{j}"](x[i])

        return tuple(out_weights)
