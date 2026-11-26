"""Check supported tensor layouts and reject unintended broadcasting."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch.nn.functional import scaled_dot_product_attention

from slm.functional import apply_rope, attention, rotate_half


@pytest.mark.parametrize("multi_head", [False, True])
@pytest.mark.parametrize("shared_mask", [False, True])
def test_attention_explicit_layouts(multi_head: bool, shared_mask: bool) -> None:
    leading = (2, 4) if multi_head else (2,)
    q = torch.randn(*leading, 3, 8)
    k = torch.randn(*leading, 5, 8)
    v = torch.randn(*leading, 5, 6)
    mask_shape = (3, 5) if shared_mask else (1,) * len(leading) + (3, 5)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    mask[..., -1] = False

    result, weights = attention(q, k, v, mask=mask)

    reference = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(result, reference)
    assert result.shape == (*leading, 3, 6)
    assert weights.shape == (*leading, 3, 5)


@pytest.mark.parametrize("argument", ["query", "key", "value"])
@pytest.mark.parametrize("axis", [0, 1])
def test_attention_rejects_batch_or_head_broadcasting(argument: str, axis: int) -> None:
    tensors = {
        "query": torch.randn(2, 4, 3, 8),
        "key": torch.randn(2, 4, 5, 8),
        "value": torch.randn(2, 4, 5, 6),
    }
    tensors[argument] = tensors[argument].narrow(axis, 0, 1)
    with pytest.raises(TypeCheckError):
        attention(tensors["query"], tensors["key"], tensors["value"])


def test_attention_rejects_mixed_ranks() -> None:
    # These shapes would otherwise broadcast successfully into a 4D result.
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 6)
    with pytest.raises(ValueError, match="same rank"):
        attention(q, k, v)


def test_attention_requires_explicit_head_axis_in_batched_mask() -> None:
    q = torch.randn(2, 2, 3, 8)
    # A 3D mask is ambiguous when batch size happens to equal the head count.
    with pytest.raises(ValueError, match="mask"):
        attention(q, q, q, mask=torch.ones(2, 3, 3))


@pytest.mark.parametrize("shape", [(2, 4, 6, 6), (2, 4, 5, 8, 6)])
def test_attention_rejects_invalid_value_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(TypeCheckError):
        attention(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 5, 8), torch.randn(shape))


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8), (2, 4, 3, 8)])
def test_rotate_half_explicit_layouts(shape: tuple[int, ...]) -> None:
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    expected = torch.cat((-x[..., 4:], x[..., :4]), dim=-1)
    torch.testing.assert_close(rotate_half(x), expected)


@pytest.mark.parametrize("shape", [(), (2, 2, 4, 3, 8)])
def test_rotate_half_rejects_other_ranks(shape: tuple[int, ...]) -> None:
    with pytest.raises(TypeCheckError):
        rotate_half(torch.randn(shape))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_rope_preserves_query_and_kv_heads(kv_heads: int) -> None:
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, kv_heads, 3, 8)
    angles = torch.randn(1, 3, 4).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()

    q_result, k_result = apply_rope(q, k, cos, sin)

    for original, result in ((q, q_result), (k, k_result)):
        rotated = torch.cat((-original[..., 4:], original[..., :4]), dim=-1)
        expected = original * cos[:, None] + rotated * sin[:, None]
        assert result.shape == original.shape
        torch.testing.assert_close(result, expected)


def test_rope_per_head_rotations() -> None:
    # Unequal sequence/head sizes ensure the annotation selects the right axis.
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn_like(q)
    cos = torch.zeros(1, 4, 8)
    sin = torch.ones_like(cos)
    q_result, k_result = apply_rope(q, k, cos, sin, unsqueeze_dim=2)
    torch.testing.assert_close(q_result, torch.cat((-q[..., 4:], q[..., :4]), dim=-1))
    torch.testing.assert_close(k_result, torch.cat((-k[..., 4:], k[..., :4]), dim=-1))


@pytest.mark.parametrize(
    "argument, shape",
    [
        ("q", (2, 1, 4, 3, 8)),
        ("k", (1, 2, 3, 8)),
        ("k", (2, 2, 1, 8)),
        ("k", (2, 2, 3, 4)),
        ("cos", (1, 1, 3, 8)),
        ("cos", (1, 5, 8)),
        ("cos", (1, 3, 1)),
    ],
)
def test_rope_rejects_invalid_shapes(argument: str, shape: tuple[int, ...]) -> None:
    tensors = {
        "q": torch.randn(2, 4, 3, 8),
        "k": torch.randn(2, 2, 3, 8),
        "cos": torch.ones(1, 3, 8),
        "sin": torch.zeros(1, 3, 8),
    }
    tensors[argument] = torch.randn(shape)
    with pytest.raises(TypeCheckError):
        apply_rope(tensors["q"], tensors["k"], tensors["cos"], tensors["sin"])


def test_rope_rejects_per_head_rotation_that_expands_kv_heads() -> None:
    with pytest.raises(ValueError, match="both query and key heads"):
        apply_rope(
            torch.randn(2, 4, 3, 8), torch.randn(2, 1, 3, 8),
            torch.ones(1, 4, 8), torch.zeros(1, 4, 8), unsqueeze_dim=2,
        )
