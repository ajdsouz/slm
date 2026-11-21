"""Unit and integration tests for the SLM neural network layers."""

from dataclasses import replace

import pytest
import torch

from slm.config import ModelConfig
from slm.layers import Attention, DecoderLayer, MLP, RMSNorm, RoPE


ModelInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
]


@pytest.fixture
def config() -> ModelConfig:
    """A small model that still exercises grouped-query attention."""
    return ModelConfig(
        block_size=8,
        d_model=8,
        d_ffn=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_positional_embed=8,
        rmsnorm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        vocab_size=32,
        activation="silu",
        pad_token_id=0,
        tie_word_embeddings=False,
    )


@pytest.fixture
def model_inputs(
    config: ModelConfig,
) -> ModelInputs:
    """Input, padding mask and RoPE embeddings for a small batch."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, config.d_model, generator=generator)
    mask = torch.ones(2, 4, dtype=torch.bool)
    positions = torch.arange(4).unsqueeze(0).expand(2, -1)
    cos, sin = RoPE(config)(x, positions)
    return x, mask, (cos, sin)


# RMSNorm
def test_rmsnorm_known_values() -> None:
    layer = RMSNorm(hidden_size=4, eps=1e-6)
    with torch.no_grad():
        layer.weight.fill_(2.0)

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = 2.0 * x / torch.sqrt(torch.tensor(7.5 + 1e-6))

    torch.testing.assert_close(layer(x), expected)


def test_rmsnorm_zero_input() -> None:
    layer = RMSNorm(hidden_size=8)
    result = layer(torch.zeros(2, 3, 8))

    torch.testing.assert_close(result, torch.zeros(2, 3, 8))
    assert torch.isfinite(result).all()


def test_rmsnorm_backward() -> None:
    layer = RMSNorm(hidden_size=8)
    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    x.requires_grad_()

    layer(x).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


# MLP
def test_mlp_known_weights(config: ModelConfig) -> None:
    # Identity projections and ReLU: output = x * ReLU(x)
    small_config = replace(config, d_ffn=8, activation="relu")
    layer = MLP(small_config)
    with torch.no_grad():
        for projection in (layer.up_proj, layer.gate_proj, layer.down_proj):
            projection.weight.copy_(torch.eye(8))

    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0, 2.0, -3.0, 4.0, -5.0]]])
    expected = torch.tensor([[[1.0, 0.0, 9.0, 0.0, 4.0, 0.0, 16.0, 0.0]]])

    torch.testing.assert_close(layer(x), expected)


def test_mlp_shape_and_backward(config: ModelConfig) -> None:
    layer = MLP(config)
    x = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8) / 64
    x.requires_grad_()

    result = layer(x)
    assert result.shape == x.shape
    result.square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# RoPE
def test_rope_known_positions(config: ModelConfig) -> None:
    layer = RoPE(config)
    x = torch.zeros(1, 2, 8)
    position_ids = torch.tensor([[0, 1]])
    cos, sin = layer(x, position_ids)
    assert cos.shape == (1, 2, 4)
    assert sin.shape == (1, 2, 4)
    torch.testing.assert_close(cos[0, 0], torch.ones(4))
    torch.testing.assert_close(sin[0, 0], torch.zeros(4))
    # head_dim=4, theta=10000 -> frequencies 1.0 and 0.01
    angles = torch.tensor([1.0, 0.01, 1.0, 0.01])
    torch.testing.assert_close(cos[0, 1], angles.cos())
    torch.testing.assert_close(sin[0, 1], angles.sin())


def test_rope_preserves_input_dtype(config: ModelConfig) -> None:
    layer = RoPE(config)
    x = torch.zeros(2, 3, 8, dtype=torch.float64)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)
    cos, sin = layer(x, positions)
    assert cos.shape == sin.shape == (2, 3, 4)
    assert cos.dtype == sin.dtype == x.dtype

# Attention
def test_attention_layer_shape_and_gqa(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    layer = Attention(config, layer_idx=0)

    assert layer.n_heads == 2
    assert layer.n_kv_heads == 1
    assert layer.n_groups == 2
    assert layer.q_proj.out_features == 8
    assert layer.k_proj.out_features == 4
    assert layer.v_proj.out_features == 4

    result = layer(x, mask, embeddings)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test_attention_layer_causal_and_padding_mask(config: ModelConfig) -> None:
    # All Q/K scores are zero, so attention is uniform among allowed keys.
    layer = Attention(config, layer_idx=0)
    with torch.no_grad():
        for projection in (layer.q_proj, layer.k_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        layer.v_proj.weight.zero_()
        layer.v_proj.weight[:, :4].copy_(torch.eye(4))
        layer.v_proj.bias.zero_()
        layer.o_proj.weight.copy_(torch.eye(8))

    x = torch.tensor([[
        [1., 2., 3., 4., 0., 0., 0., 0.],
        [5., 6., 7., 8., 0., 0., 0., 0.],
        [9., 10., 11., 12., 0., 0., 0., 0.],
    ]])
    mask = torch.tensor([[True, True, False]])
    positions = torch.arange(3).unsqueeze(0)
    embeddings = RoPE(config)(x, positions)

    result = layer(x, mask, embeddings)
    expected = torch.tensor([[
        [1., 2., 3., 4., 1., 2., 3., 4.],
        [3., 4., 5., 6., 3., 4., 5., 6.],
        [3., 4., 5., 6., 3., 4., 5., 6.],
    ]])
    torch.testing.assert_close(result, expected)


def test_attention_layer_cannot_see_future(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    layer = Attention(config, layer_idx=0)
    changed = x.clone()
    changed[:, -1, :] += 100.0

    original_output = layer(x, mask, embeddings)
    changed_output = layer(changed, mask, embeddings)

    # Changing the last token must not affect earlier positions.
    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])


def test_attention_layer_backward(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    x = x.clone().detach().requires_grad_()
    layer = Attention(config, layer_idx=0)

    layer(x, mask, embeddings).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# DecoderLayer
def test_decoder_layer_forward(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    layer = DecoderLayer(config, layer_idx=0)

    result = layer(x, mask=mask, position_embedding=embeddings)

    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test_decoder_layer_residual_identity(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    layer = DecoderLayer(config, layer_idx=0)

    # Force both residual branches to output zero.
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()

    result = layer(x, mask=mask, position_embedding=embeddings)
    torch.testing.assert_close(result, x)


def test_decoder_layer_backward(config: ModelConfig, model_inputs: ModelInputs) -> None:
    x, mask, embeddings = model_inputs
    x = x.clone().detach().requires_grad_()
    layer = DecoderLayer(config, layer_idx=0)

    result = layer(x, mask=mask, position_embedding=embeddings)
    result.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
