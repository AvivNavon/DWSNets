from typing import Optional, Tuple

import torch
from torch.nn import ModuleDict

from nn.layers.base import BaseLayer, GeneralSetLayer


class GeneralMatrixSetLayer(BaseLayer):
    """General matrix set layer."""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        in_index,
        out_index,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "sab",
        first_dim_is_input=False,
        last_dim_is_input=False,
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
        :param num_heads:
        :param set_layer:
        :param first_dim_is_input: indicates that the input matrix (of in_shapes) is the weight matrix for the
            first layer (of the input net, e.g. INR).
        :param last_dim_is_input: indicates that the output matrix is the weight matrix for the first layer.
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
            num_heads=num_heads,
            set_layer=set_layer,
        )
        self.first_dim_is_input = first_dim_is_input
        self.last_dim_is_input = last_dim_is_input
        self.first_dim_is_output = first_dim_is_output
        self.last_dim_is_output = last_dim_is_output

        self.in_index = in_index
        self.out_index = out_index

        # todo: we can greatly reduce the number of if else if we will use the feature_index
        if in_index == out_index:
            assert not (first_dim_is_input and last_dim_is_input)
            self.feature_index = (
                0 if first_dim_is_input else 1
            )  # 0 means we are at first layer, 1 means last layer
            # this is the case we map W_i to W_i where W_i is the first or last layer's weight matrix
            in_features = in_features * in_shape[self.feature_index]
            out_features = out_features * in_shape[self.feature_index]

        elif in_index == out_index - 1:
            # this is the case we map W_i to W_j where i=j-1
            assert not (first_dim_is_input and last_dim_is_output)
            if first_dim_is_input:
                # i=0 and j=1
                self.feature_index = 0
                in_features = in_features * in_shape[self.feature_index]
                out_features = out_features
            elif last_dim_is_output:
                # i=L-2 and j=L-1
                self.feature_index = 1
                in_features = in_features
                out_features = out_features * out_shape[self.feature_index]
            else:
                # internal layers
                in_features = in_features
                out_features = out_features

        else:
            # i = j + 1
            assert in_index == out_index + 1  # in_shape[0] == out_shape[-1]
            assert not (last_dim_is_input and first_dim_is_output)
            if last_dim_is_input:
                # j=0, i=1
                self.feature_index = 0
                in_features = in_features
                out_features = out_features * out_shape[self.feature_index]

            elif first_dim_is_output:
                # j=L-2, i=L-1
                self.feature_index = 1
                in_features = in_features * in_shape[self.feature_index]
                out_features = out_features

            else:
                # internal layers
                in_features = in_features
                out_features = out_features

        self.set_layer = GeneralSetLayer(
            in_features=in_features,
            out_features=out_features,
            reduction=reduction,
            bias=bias,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
        )

    def forward(self, x):
        if self.in_index == self.out_index:
            # this is the case we map W_i to W_i where W_i is the first or last layer's weight matrix
            if self.first_dim_is_input:
                # first layer, feature_index is d0
                # (bs, d1, d0, in_features)
                x = x.permute(0, 2, 1, 3)

            # (bs, set_dim, feature_dim * in_features)
            x = x.flatten(start_dim=2)
            # (bs, set_dim, feature_dim * out_features)
            x = self.set_layer(x)
            # (bs, set_dim, feature_dim, out_features)
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.in_shape[self.feature_index],
                self.out_features,
            )

            if self.first_dim_is_input:
                # permute back to (bs, d0, d1, out_features)
                x = x.permute(0, 2, 1, 3)

        elif (
            self.in_index == self.out_index - 1
        ):  # self.in_shape[-1] == self.out_shape[0]:
            # i -> j  where i=j-1
            if self.first_dim_is_input:
                # i=0 and j=1
                # (bs, d1, d0 * in_features)
                x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
                # (bs, d1, out_features)
                x = self.set_layer(x)
                # (bs, d1, d2, out_features)
                x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)

            elif self.last_dim_is_output:
                # i=L-2 and j=L-1
                # (bs, d_{L-2}, d_{L-1}, in_features)
                # (bs, d_{L-1}, in_features)
                x = self._reduction(x, dim=1)
                # (bs, d_{L-1}, d_L * out_features)
                x = self.set_layer(x)
                # (bs, d_{L-1}, d_L, out_features)
                x = x.reshape(x.shape[0], *self.out_shape, self.out_features)
            else:
                # internal layers
                # (bs, d_i, d_{i+1}, in_features)
                # (bs, d_{i+1}, in_features)
                x = self._reduction(x, dim=1)
                # (bs, d_{i+1}, out_features)
                x = self.set_layer(x)
                # (bs, d_{i+1}, d_{i+2}, out_features)
                x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)

        else:
            # i = j + 1
            if self.last_dim_is_input:
                # i=1, j=0
                # (bs, d1, d2, in_features)
                # (bs, d1, in_features)
                x = self._reduction(x, dim=2)
                # (bs, d1, d0 * out_features)
                x = self.set_layer(x)
                # (bs, d1, d0, out_features)
                x = x.reshape(
                    x.shape[0], x.shape[1], self.out_shape[0], self.out_features
                )
                # (bs, d0, d1, out_features)
                x = x.permute(0, 2, 1, 3)

            elif self.first_dim_is_output:
                # i=L-1, j=L-2
                # (bs, d_{L-1}, d_L, in_features)
                # (bs, d_{L-1}, out_features)
                x = self.set_layer(x.flatten(start_dim=2))
                x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1, 1)

            else:
                # internal layers (j = i-1):
                # (bs, d_i, d_{i+1}, in_feature) -> (bs, d_{i-1}, d_i, out_features)
                # (bs, d_i, in_feature)
                x = self._reduction(x, dim=2)
                # (bs, d_i, out_feature)
                x = self.set_layer(x)
                # (bs, d_{i-1}, d_i, out_feature)
                x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1, 1)

        return x


class SetKroneckerSetLayer(BaseLayer):
    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        reduction="max",
        bias=True,
        n_fc_layers=1,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            in_shape=in_shape,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
            bias=bias,
        )
        # todo: bias is overparametrized here. we can reduce the number of parameters
        self.d1, self.d2 = in_shape
        self.in_features = in_features

        self.lin_all = self._get_mlp(in_features, out_features, bias=bias)
        self.lin_n = self._get_mlp(in_features, out_features, bias=bias)
        self.lin_m = self._get_mlp(in_features, out_features, bias=bias)
        self.lin_both = self._get_mlp(in_features, out_features, bias=bias)

        # todo: add attention support
        # if reduction == "attn":
        #     self.attn0 = Attn(self.d2 * self.in_features)
        #     self.attn1 = Attn(self.d1 * self.in_features)
        #     self.attn2 = Attn(self.in_features)

    def forward(self, x):
        # x is [b, d1, d2, f]
        shapes = x.shape
        bs = shapes[0]
        # all
        out_all = self.lin_all(x)  # [b, d1, d2, f] -> [b, d1, d2, f']
        # rows
        pooled_rows = self._reduction(
            x, dim=1, keepdim=True
        )  # [b, d1, d2, f] -> [b, 1, d2, f]
        out_rows = self.lin_n(pooled_rows)  # [b, 1, d2, f] -> [b, 1, d2, f']
        # cols
        pooled_cols = self._reduction(
            x, dim=2, keepdim=True
        )  # [b, d1, d2, f] -> [b, d1, 1, f]
        out_cols = self.lin_m(pooled_cols)  # [b, d1, 1, f] -> [b, d1, 1, f']
        # both
        # todo: need to understand how we do this generic enough to move it into self._reduction.
        #  I think we can just flatten (1, 2) and call it on the flat axis
        # if self.reduction == "max":
        #     pooled_all, _ = torch.max(
        #         x.permute(0, 3, 1, 2).flatten(start_dim=2), dim=-1, keepdim=True
        #     )
        #     pooled_all = pooled_all.permute(0, 2, 1).unsqueeze(
        #         1
        #     )  # [b, d1, d2, f] -> [b, 1, 1, f]
        # else:
        # pooled_all = self._reduction(x, dim=(1, 2), keepdim=True)
        x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
        pooled_all = self._reduction(x, dim=2)
        pooled_all = pooled_all.unsqueeze(1).unsqueeze(
            1
        )  # [b, d1, d2, f] -> [b, 1, 1, f]

        out_both = self.lin_both(pooled_all)  # [b, 1, 1, f] -> [b, 1, 1, f']

        new_features = (
            out_all + out_rows + out_cols + out_both
        ) / 4.0  # [b, d1, d2, f']
        return new_features


class FromFirstLayer(BaseLayer):
    """Mapping W_0 -> W_j where j != 1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
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
        self.last_dim_is_output = last_dim_is_output

        if self.last_dim_is_output:
            # i=0, j=L-1
            in_features = self.in_features * self.in_shape[0]  # d0 * in_features
            out_features = self.out_features * self.out_shape[1]  # dL * out_features
            self.layer = self._get_mlp(
                in_features=in_features, out_features=out_features, bias=bias
            )

        else:
            # i=0, j != L-1
            in_features = self.in_features * self.in_shape[0]  # d0 * in_features
            out_features = self.out_features  # out_features
            self.layer = self._get_mlp(
                in_features=in_features, out_features=out_features, bias=bias
            )

    def forward(self, x):
        if self.last_dim_is_output:
            # i=0, j=L-1
            # (bs, d0, d1, in_features)
            # (bs, d0, in_features)
            x = self._reduction(x, dim=2)
            # (bs, dL * out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, d_{L-1}, dL, out_features)
            x = (
                x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
                .unsqueeze(1)
                .repeat(1, self.out_shape[0], 1, 1)
            )
        else:
            # i=0, j != L-1
            # (bs, d0, d1, in_features)
            # (bs, d0, in_features)
            x = self._reduction(x, dim=2)
            # (bs, out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, d_j, d_{j+1}, out_features)
            x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
        return x


class ToFirstLayer(BaseLayer):
    """Mapping W_i -> W_0 where i != 1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
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
        self.first_dim_is_output = first_dim_is_output

        if self.first_dim_is_output:
            # i=L-1, j=0
            in_features = self.in_features * self.in_shape[-1]  # dL * in_features
            out_features = self.out_features * self.out_shape[0]  # d0 * out_features
            self.layer = self._get_mlp(in_features, out_features, bias=bias)

        else:
            # i!=L-1, j=0
            in_features = self.in_features  # in_features
            out_features = self.out_features * self.out_shape[0]  # d0 * out_features
            self.layer = self._get_mlp(in_features, out_features, bias=bias)

    def forward(self, x):
        if self.first_dim_is_output:
            # i=L-1, j=0
            # (bs, d{L-1}, dL, in_features)
            # (bs, dL, in_features)
            x = self._reduction(x, dim=1)
            # (bs, d0 * out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, d0, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
            # (bs, d0, d1, out_features)
            x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)
        else:
            # (bs, dj, d{j+1}, in_features)
            # (bs, in_features, dj * d{j+1})
            x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
            # (bs, in_features)
            x = self._reduction(x, dim=2)
            # (bs, d0 * out_features)
            x = self.layer(x.flatten(start_dim=1))
            # (bs, d0, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
            # (bs, d0, d1, out_features)
            x = x.unsqueeze(2).repeat(1, 1, self.out_shape[-1], 1)
        return x


class FromLastLayer(BaseLayer):
    """Mapping W_{L-1} -> W_j where j != 0, L-2"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
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
        self.layer = self._get_mlp(
            in_features=in_features * self.in_shape[-1],  # dL * in_features
            out_features=out_features,  # out_features
            bias=bias,
        )

    def forward(self, x):
        # (bs, d{L-1}, dL, in_features)
        # (bs, dL, in_features)
        x = self._reduction(x, dim=1)
        # (bs, out_features)
        x = self.layer(x.flatten(start_dim=1))
        # (bs, *out_shape, out_features)
        x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
        return x


class ToLastLayer(BaseLayer):
    """Mapping W_i -> W_{L-1} where i != 0, L-2"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
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
        self.layer = self._get_mlp(
            in_features=in_features,  # dL * in_features
            out_features=out_features * self.out_shape[-1],  # out_features * dL
            bias=bias,
        )

    def forward(self, x):
        # (bs, di, d{i+1}, in_features)
        # (bs, in_features, di * d{i+1})
        x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
        # (bs, in_features)
        x = self._reduction(x, dim=2)
        # (bs, dL * out_features)
        x = self.layer(x)
        # (bs, dL, out_features)
        x = x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
        # (bs, d{L-1}, dL, out_features)
        x = x.unsqueeze(1).repeat(1, self.out_shape[0], 1, 1)
        return x


class NonNeighborInternalLayer(BaseLayer):
    """Mapping W_i -> W_j where i,j != 0, L-2 and |i-j|>1"""

    def __init__(
        self,
        in_features,
        out_features,
        in_shape,
        out_shape,
        bias: bool = True,
        reduction: str = "max",
        n_fc_layers: int = 1,
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

        self.layer = self._get_mlp(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        # (bs, di, d{i+1}, in_features)
        # (bs, in_features, di * d{i+1})
        x = x.permute(0, 3, 1, 2).flatten(start_dim=2)
        # (bs, in_features)
        x = self._reduction(x, dim=2)
        # (bs, out_features)
        x = self.layer(x)
        # (bs, *out_shape, out_features)
        x = x.unsqueeze(1).unsqueeze(1).repeat(1, *self.out_shape, 1)
        return x


class WeightToWeightBlock(BaseLayer):
    def __init__(
        self,
        in_features,
        out_features,
        shapes,
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
        assert all([len(shape) == 2 for shape in shapes])
        assert len(shapes) > 2

        self.shapes = shapes
        self.n_layers = len(shapes)
        self.diagonal = diagonal

        self.layers = ModuleDict()
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and abs(i - j) > 1:
                    continue
                if i == j:
                    if i == 0:
                        # W0 -> W0
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            first_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif j == self.n_layers - 1:
                        # W{L-1} -> W{L-1}
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        self.layers[f"{i}_{j}"] = SetKroneckerSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                        )

                elif i == j - 1:
                    if i == 0:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            first_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif j == self.n_layers - 1:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            in_index=i,
                            out_index=j,
                        )
                elif i == j + 1:
                    if j == 0:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif i == self.n_layers - 1:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            first_dim_is_output=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        self.layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=in_features,
                            out_features=out_features,
                            in_shape=shapes[i],
                            out_shape=shapes[j],
                            reduction=reduction,
                            bias=bias,
                            num_heads=num_heads,
                            set_layer=set_layer,
                            n_fc_layers=n_fc_layers,
                            in_index=i,
                            out_index=j,
                        )
                elif i == 0:
                    self.layers[f"{i}_{j}"] = FromFirstLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                        last_dim_is_output=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif j == 0:
                    self.layers[f"{i}_{j}"] = ToFirstLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif i == self.n_layers - 1:
                    # j != i-1, 0
                    self.layers[f"{i}_{j}"] = FromLastLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                    )
                elif j == self.n_layers - 1:
                    self.layers[f"{i}_{j}"] = ToLastLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                    )
                else:
                    assert abs(i - j) > 1
                    self.layers[f"{i}_{j}"] = NonNeighborInternalLayer(
                        in_features=in_features,
                        out_features=out_features,
                        in_shape=shapes[i],
                        out_shape=shapes[j],
                        reduction=reduction,
                        bias=bias,
                        n_fc_layers=n_fc_layers,
                    )

    def forward(self, x: Tuple[torch.tensor]):
        out_weights = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and abs(i - j) > 1:
                    continue
                out_weights[j] = out_weights[j] + self.layers[f"{i}_{j}"](x[i])
        return tuple(out_weights)


if __name__ == "__main__":
    d0, d1, d2, d3, d4, d5 = 2, 10, 20, 30, 40, 1
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    print(len(matrices))
    weight_block = WeightToWeightBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:3] for m in matrices)
    )
    out = weight_block(matrices)
    print([o.shape for o in out])

    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = weight_block(
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        )
    )

    assert torch.allclose(out[0][:, :, perm1, :], out_perm[0], atol=1e-5, rtol=0)
    assert torch.allclose(
        out[1][:, perm1, :, :][:, :, perm2, :], out_perm[1], atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out[2][:, perm2, :, :][:, :, perm3, :], out_perm[2], atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out[3][:, perm3, :, :][:, :, perm4, :], out_perm[3], atol=1e-5, rtol=0
    )
    assert torch.allclose(out[4][:, perm4, :, :], out_perm[4], atol=1e-5, rtol=0)
