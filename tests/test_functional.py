import pytest
import torch

from slm.functional import (
    rotate_half,
    split_heads,
    merge_heads,
    repeat_kv,
    apply_rope,
    maybe_autocast,
)

# rotate_half
def test_rotate_half_values() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    result = rotate_half(x)
    torch.testing.assert_close(result, expected)


def test_rotate_half_last_dimension() -> None:
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    expected = torch.tensor([
        [-3.0, -4.0, 1.0, 2.0],
        [-7.0, -8.0, 5.0, 6.0],
    ])
    result = rotate_half(x)
    assert result.shape == x.shape
    torch.testing.assert_close(result, expected)


# split_heads
def test_split_heads_values() -> None:
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    ])
    expected = torch.tensor([
        [
            [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]],
            [[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]],
        ]
    ])
    result = split_heads(x, n_heads=2)
    assert result.shape == (1, 2, 3, 2)
    torch.testing.assert_close(result, expected)


# merge_heads
def test_merge_heads_values() -> None:
    x = torch.tensor([
        [
            [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]],
            [[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]],
        ]
    ])
    expected = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    ])
    result = merge_heads(x)
    assert result.shape == (1, 3, 4)
    torch.testing.assert_close(result, expected)


def test_split_merge_roundtrip() -> None:
    x = torch.arange(
        48, dtype=torch.float32
    ).reshape(2, 3, 8)
    split = split_heads(x, n_heads=2)
    result = merge_heads(split)
    assert split.shape == (2, 2, 3, 4)
    torch.testing.assert_close(result, x)


# repeat_kv
@pytest.mark.parametrize("n_rep", [1, 2, 3])
def test_repeat_kv_values(n_rep: int) -> None:
    x = torch.arange(
        24, dtype=torch.float32
    ).reshape(2, 2, 3, 2)
    expected = torch.repeat_interleave(
        x, repeats=n_rep, dim=1
    )
    result = repeat_kv(x, n_rep=n_rep)
    assert result.shape == (2, 2 * n_rep, 3, 2)
    torch.testing.assert_close(result, expected)

# apply_rope
def test_apply_rope_identity() -> None:
    # No rotation: cos = 1, sin = 0
    q = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    k = q + 10.0

    cos = torch.ones(1, 2, 4)
    sin = torch.zeros(1, 2, 4)

    q_result, k_result = apply_rope(q, k, cos, sin)

    torch.testing.assert_close(q_result, q)
    torch.testing.assert_close(k_result, k)


def test_apply_rope_quarter_turn() -> None:
    # 90-degree rotation: cos = 0, sin = 1
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])

    cos = torch.zeros(1, 1, 4)
    sin = torch.ones(1, 1, 4)

    q_result, k_result = apply_rope(q, k, cos, sin)

    expected_q = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    expected_k = torch.tensor([[[[-7.0, -8.0, 5.0, 6.0]]]])

    torch.testing.assert_close(q_result, expected_q)
    torch.testing.assert_close(k_result, expected_k)


def test_apply_rope_unsqueeze_dim_two() -> None:
    # Different rotation behavior for each attention head
    q = torch.tensor([[
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        [[5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]],
    ]])
    k = q * 2

    # Head 1 rotates, Head 2 stays unchanged
    cos = torch.tensor([
        [[0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0]]
    ])
    sin = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]]
    ])

    q_result, k_result = apply_rope(
        q, k, cos, sin, unsqueeze_dim=2
    )

    expected_q = torch.tensor([[
        [[-3.0, -4.0, 1.0, 2.0],
         [-3.0, -4.0, 1.0, 2.0]],
        [[5.0, 6.0, 7.0, 8.0],
         [5.0, 6.0, 7.0, 8.0]],
    ]])

    torch.testing.assert_close(q_result, expected_q)
    torch.testing.assert_close(k_result, expected_q * 2)

# maybe_autocast
def test_maybe_autocast_disabled() -> None:
    # Start with autocast disabled
    with torch.autocast(device_type="cpu", enabled=False):
        with maybe_autocast(
            "cpu", dtype=torch.bfloat16, enabled=False
        ):
            assert not torch.is_autocast_enabled("cpu")

def test_maybe_autocast_enabled() -> None:
    # Autocast should turn on temporarily
    with torch.autocast(device_type="cpu", enabled=False):
        with maybe_autocast(
            "cpu", dtype=torch.bfloat16, enabled=True
        ):
            assert torch.is_autocast_enabled("cpu")

        # Leaving the context restores the original state
        assert not torch.is_autocast_enabled("cpu")

def test_maybe_autocast_nested_disabled() -> None:
    # Autocast is initially enabled
    with torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        assert torch.is_autocast_enabled("cpu")

        # Temporarily disable it inside another context
        with maybe_autocast("cpu", enabled=False):
            assert not torch.is_autocast_enabled("cpu")

        # The outer context is still enabled
        assert torch.is_autocast_enabled("cpu")


