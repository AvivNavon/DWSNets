import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BaseLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        in_shape: Optional[Tuple] = None,
        out_shape: Optional[Tuple] = None,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
        num_heads: int = 8,
        set_layer: str = "ds",
    ):
        super().__init__()
        assert set_layer in ["ds", "sab"]

        self.in_features = in_features
        self.out_features = out_features
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bias = bias
        assert reduction in ["mean", "sum", "attn", "max"]
        self.reduction = reduction
        self.b = None
        self.n_fc_layers = n_fc_layers
        self.num_heads = num_heads

    def _get_mlp(self, in_features, out_features, bias=False):
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(self.n_fc_layers - 1):
            layers.extend([nn.ReLU(), nn.Linear(out_features, out_features, bias=bias)])
        return nn.Sequential(*layers)

    def _init_bias(self, row_equal, col_equal, row_dim, col_dim):
        if self.bias:
            b = torch.empty(
                1 if row_equal else row_dim,
                1 if col_equal else col_dim,
                self.out_features,
            )
            b.uniform_(-1e-2, 1e-2)
            self.b = nn.Parameter(b)

    def _reduction(self, x: torch.tensor, dim=1, keepdim=False):
        if self.reduction == "mean":
            x = x.mean(dim=dim, keepdim=keepdim)
        elif self.reduction == "sum":
            x = x.sum(dim=dim, keepdim=keepdim)
        elif self.reduction == "attn":
            assert x.ndim == 3
            raise NotImplementedError
        elif self.reduction == "max":
            x, _ = torch.max(x, dim=dim, keepdim=keepdim)
        else:
            raise ValueError(f"invalid reduction, got {self.reduction}")
        return x


class MAB(nn.Module):
    """https://github.com/juho-lee/set_transformer/blob/master/modules.py"""

    # todo: check bias here
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(BaseLayer):
    def __init__(self, in_features, out_features, num_heads=8, ln=False):
        super().__init__(in_features, out_features)
        self.mab = MAB(in_features, in_features, out_features, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class SetLayer(BaseLayer):
    """
    from https://github.com/manzilzaheer/DeepSets/tree/master/PointClouds
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            reduction=reduction,
            n_fc_layers=n_fc_layers,
        )
        self.Gamma = self._get_mlp(in_features, out_features, bias=self.bias)
        self.Lambda = self._get_mlp(in_features, out_features, bias=False)
        self.reduction = reduction
        if self.reduction == "attn":
            self.attn = Attn(dim=in_features)

    def forward(self, x):
        # set dim is 1
        if self.reduction == "mean":
            xm = x.mean(1, keepdim=True)
        elif self.reduction == "sum":
            xm = x.sum(1, keepdim=True)
        elif self.reduction == "attn":
            xm = self.attn(x.transpose(-1, -2), keepdim=True).transpose(-1, -2)
        else:
            xm, _ = torch.max(x, dim=1, keepdim=True)

        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class GeneralSetLayer(BaseLayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        reduction: str = "mean",
        n_fc_layers: int = 1,
        num_heads=8,
        set_layer="ds",
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
        self.set_layer = dict(
            ds=SetLayer(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                reduction=reduction,
                n_fc_layers=n_fc_layers,
            ),
            sab=SAB(
                in_features=in_features, out_features=out_features, num_heads=num_heads
            ),
        )[set_layer]

    def forward(self, x):
        return self.set_layer(x)


class Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Parameter(
            torch.ones(
                dim,
            )
        )

    def forward(self, x, keepdim=False):
        # Note: reduction is applied to last dim. For example for (bs, d, d') we compute d' attn weights
        # by multiplying over d.
        attn = (x.transpose(-1, -2) * self.query).sum(-1)
        attn = F.softmax(attn, dim=-1)
        # todo: change to attn.unsqueeze(-2) ?
        if x.ndim == 3:
            attn = attn.unsqueeze(1)
        elif x.ndim == 4:
            attn = attn.unsqueeze(2)

        output = (x * attn).sum(-1, keepdim=keepdim)

        return output
