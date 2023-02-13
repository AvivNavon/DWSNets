import torch

from experiments.utils import set_seed
from nn.layers.bias_to_bias import BiasToBiasBlock
from nn.layers.bias_to_weight import BiasToWeightBlock
from nn.layers.weight_to_bias import WeightToBiasBlock
from nn.layers.weight_to_weight import (
    FromFirstLayer,
    FromLastLayer,
    NonNeighborInternalLayer,
    ToFirstLayer,
    ToLastLayer,
    WeightToWeightBlock,
)
from nn.models import DWSModel, DWSModelForClassification

set_seed(42)


def test_w_t_w_from_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = FromFirstLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[0],
        out_shape=shapes[-1],
        last_dim_is_output=True,
    )

    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)

    out_perm = layer(matrices[0][:, :, perm1, :])
    out = layer(matrices[0])
    assert torch.allclose(out[:, perm4, :, :], out_perm, atol=1e-5, rtol=0)


def test_w_t_w_to_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = ToFirstLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[-1],
        out_shape=shapes[0],
        first_dim_is_output=True,
    )

    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)

    out_perm = layer(matrices[-1][:, perm4, :, :])
    out = layer(matrices[-1])
    assert torch.allclose(out[:, :, perm1, :], out_perm, atol=1e-5, rtol=0)


def test_w_t_w_from_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = FromLastLayer(
        in_features=12, out_features=24, in_shape=shapes[-1], out_shape=shapes[2]
    )

    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)

    out_perm = layer(matrices[-1][:, perm4, :, :])
    out = layer(matrices[-1])
    assert torch.allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, atol=1e-5, rtol=0
    )


def test_w_t_w_to_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = ToLastLayer(
        in_features=12, out_features=24, in_shape=shapes[2], out_shape=shapes[-1]
    )

    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)

    out_perm = layer(matrices[2][:, perm2, :, :][:, :, perm3, :])
    out = layer(matrices[2])
    assert torch.allclose(out[:, perm4, :, :], out_perm, atol=1e-5, rtol=0)


def test_w_t_w_non_n():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = NonNeighborInternalLayer(
        in_features=12, out_features=24, in_shape=shapes[1], out_shape=shapes[3]
    )

    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)

    out_perm = layer(matrices[1][:, perm1, :, :][:, :, perm2, :])
    out = layer(matrices[1])
    assert torch.allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, atol=1e-5, rtol=0
    )


def test_weight_to_weight_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )

    weight_block = WeightToWeightBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:3] for m in matrices)
    )
    out = weight_block(matrices)

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


def test_bias_to_bias_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d1, 12),
        torch.randn(4, d2, 12),
        torch.randn(4, d3, 12),
        torch.randn(4, d4, 12),
        torch.randn(4, d5, 12),
    )

    bias_block = BiasToBiasBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:2] for m in matrices)
    )
    out = bias_block(matrices)
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = bias_block(
        (
            matrices[0][:, perm1, :],
            matrices[1][:, perm2, :],
            matrices[2][:, perm3, :],
            matrices[3][:, perm4, :],
            matrices[4],
        )
    )

    assert torch.allclose(out[0][:, perm1, :], out_perm[0], atol=1e-5, rtol=0)
    assert torch.allclose(out[1][:, perm2, :], out_perm[1], atol=1e-5, rtol=0)
    assert torch.allclose(out[2][:, perm3, :], out_perm[2], atol=1e-5, rtol=0)
    assert torch.allclose(out[3][:, perm4, :], out_perm[3], atol=1e-5, rtol=0)
    assert torch.allclose(out[4], out_perm[4], atol=1e-5, rtol=0)


def test_bias_to_weight_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d1, 12),
        torch.randn(4, d2, 12),
        torch.randn(4, d3, 12),
        torch.randn(4, d4, 12),
        torch.randn(4, d5, 12),
    )
    weights_shape = ((d0, d1), (d1, d2), (d2, d3), (d3, d4), (d4, d5))

    bias_block = BiasToWeightBlock(
        in_features=12,
        out_features=24,
        weight_shapes=weights_shape,
        bias_shapes=tuple(m.shape[1:2] for m in matrices),
    )
    out = bias_block(matrices)
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = bias_block(
        (
            matrices[0][:, perm1, :],
            matrices[1][:, perm2, :],
            matrices[2][:, perm3, :],
            matrices[3][:, perm4, :],
            matrices[4],
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


def test_weight_to_bias_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    matrices = (
        torch.randn(4, d0, d1, 12),
        torch.randn(4, d1, d2, 12),
        torch.randn(4, d2, d3, 12),
        torch.randn(4, d3, d4, 12),
        torch.randn(4, d4, d5, 12),
    )
    bias_shape = ((d1,), (d2,), (d3,), (d4,), (d5,))

    bias_block = WeightToBiasBlock(
        in_features=12,
        out_features=24,
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=bias_shape,
    )
    out = bias_block(matrices)
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = bias_block(
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        )
    )

    assert torch.allclose(out[0][:, perm1, :], out_perm[0], atol=1e-5, rtol=0)
    assert torch.allclose(out[1][:, perm2, :], out_perm[1], atol=1e-5, rtol=0)
    assert torch.allclose(out[2][:, perm3, :], out_perm[2], atol=1e-5, rtol=0)
    assert torch.allclose(out[3][:, perm4, :], out_perm[3], atol=1e-5, rtol=0)
    assert torch.allclose(out[4], out_perm[4], atol=1e-5, rtol=0)


def test_model_invariance():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    weights = (
        torch.randn(4, d0, d1, 2),
        torch.randn(4, d1, d2, 2),
        torch.randn(4, d2, d3, 2),
        torch.randn(4, d3, d4, 2),
        torch.randn(4, d4, d5, 2),
    )
    biases = (
        torch.randn(4, d1, 2),
        torch.randn(4, d2, 2),
        torch.randn(4, d3, 2),
        torch.randn(4, d4, 2),
        torch.randn(4, d5, 2),
    )

    model = DWSModelForClassification(
        input_features=2,
        n_classes=10,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        dropout_rate=0.0,
    )
    out = model((weights, biases))
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = model(
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    assert torch.allclose(out, out_perm, atol=1e-5, rtol=0)


def test_model_equivariance():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    weights = (
        torch.randn(4, d0, d1, 2),
        torch.randn(4, d1, d2, 2),
        torch.randn(4, d2, d3, 2),
        torch.randn(4, d3, d4, 2),
        torch.randn(4, d4, d5, 2),
    )
    biases = (
        torch.randn(4, d1, 2),
        torch.randn(4, d2, 2),
        torch.randn(4, d3, 2),
        torch.randn(4, d4, 2),
        torch.randn(4, d5, 2),
    )

    model = DWSModel(
        input_features=2,
        output_features=8,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        dropout_rate=0.0,
        bias=True,
    )
    out = model((weights, biases))
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = model(
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    assert torch.allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[3][:, perm3, :, :][:, :, perm4, :],
        out_weights_perm[3],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[4][:, perm4, :, :], out_weights_perm[4], atol=1e-4, rtol=0
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    assert torch.allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[3][:, perm4, :], out_biases_perm[3], atol=1e-4, rtol=0
    )
    assert torch.allclose(out_biases[4], out_biases_perm[4], atol=1e-4, rtol=0)


def test_model_equivariance_downsample():
    d0, d1, d2, d3, d4, d5 = 64, 32, 32, 32, 32, 3
    weights = (
        torch.randn(4, d0, d1, 2),
        torch.randn(4, d1, d2, 2),
        torch.randn(4, d2, d3, 2),
        torch.randn(4, d3, d4, 2),
        torch.randn(4, d4, d5, 2),
    )
    biases = (
        torch.randn(4, d1, 2),
        torch.randn(4, d2, 2),
        torch.randn(4, d3, 2),
        torch.randn(4, d4, 2),
        torch.randn(4, d5, 2),
    )

    model = DWSModel(
        input_features=2,
        output_features=8,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        input_dim_downsample=16,
        dropout_rate=0.0,
    )
    out = model((weights, biases))
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = model(
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    assert torch.allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[3][:, perm3, :, :][:, :, perm4, :],
        out_weights_perm[3],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[4][:, perm4, :, :], out_weights_perm[4], atol=1e-4, rtol=0
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    assert torch.allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[3][:, perm4, :], out_biases_perm[3], atol=1e-4, rtol=0
    )
    assert torch.allclose(out_biases[4], out_biases_perm[4], atol=1e-4, rtol=0)


def test_model_equivariance_downsample_sab():
    d0, d1, d2, d3, d4 = 28 * 28, 128, 128, 128, 10
    weights = (
        torch.randn(4, d0, d1, 1),
        torch.randn(4, d1, d2, 1),
        torch.randn(4, d2, d3, 1),
        torch.randn(4, d3, d4, 1),
    )
    biases = (
        torch.randn(4, d1, 1),
        torch.randn(4, d2, 1),
        torch.randn(4, d3, 1),
        torch.randn(4, d4, 1),
    )

    model = DWSModel(
        input_features=1,
        output_features=1,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=64,
        input_dim_downsample=16,
        dropout_rate=0.0,
        add_skip=True,
    )
    out = model((weights, biases))
    # perm test
    perm1 = torch.randperm(d1)
    perm2 = torch.randperm(d2)
    perm3 = torch.randperm(d3)
    perm4 = torch.randperm(d4)
    out_perm = model(
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    assert torch.allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        out_weights[3][:, perm3, :, :], out_weights_perm[3], atol=1e-4, rtol=0
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    assert torch.allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=0
    )
    assert torch.allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=0
    )
    assert torch.allclose(out_biases[3], out_biases_perm[3], atol=1e-4, rtol=0)
