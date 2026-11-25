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
    """Verify the rotation of a one-dimensional vector.

    Split [1, 2, 3, 4] into [1, 2] and [3, 4]. Moving the negated
    second half before the first half gives [-3, -4, 1, 2].
    """
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    result = rotate_half(x)
    torch.testing.assert_close(result, expected)


def test_rotate_half_last_dimension() -> None:
    """Verify that rotation acts independently along the last dimension.

    The rows [1, 2, 3, 4] and [5, 6, 7, 8] become [-3, -4, 1, 2]
    and [-7, -8, 5, 6], respectively, without changing the shape.
    """
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
    """Verify the shape and feature order when splitting attention heads.

    The input has shape (1, 3, 4). Splitting four features into two
    heads assigns the first two features of each token to head 0 and
    the last two to head 1. The output shape is (1, 2, 3, 2).
    """
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
    """Verify the shape and feature order when merging attention heads.

    The input has shape (1, 2, 3, 2). Concatenate the two heads for
    each token, in head order, to recover the rows [1, 2, 3, 4],
    [5, 6, 7, 8] and [9, 10, 11, 12] in shape (1, 3, 4).
    """
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
    """Verify that merging reverses the splitting operation.

    The 48 distinct values in shape (2, 3, 8) are split into two heads
    of width four, giving (2, 2, 3, 4). Merging must recover the
    original tensor, including its element order.
    """
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
    """Verify consecutive repetition of each key/value head.

    For n_rep in {1, 2, 3}, compare against torch.repeat_interleave
    along the head axis. The shape changes from (2, 2, 3, 2) to
    (2, 2 * n_rep, 3, 2). A repetition factor of one preserves values.
    """
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
    """Verify that a zero-angle rotary embedding leaves Q and K unchanged.

    For either tensor x, x * cos + rotate_half(x) * sin equals x
    when cos = 1 and sin = 0. The coefficients broadcast over heads.
    """
    # No rotation: cos = 1, sin = 0
    q = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    k = q + 10.0

    cos = torch.ones(1, 2, 4)
    sin = torch.zeros(1, 2, 4)

    q_result, k_result = apply_rope(q, k, cos, sin)

    torch.testing.assert_close(q_result, q)
    torch.testing.assert_close(k_result, k)


def test_apply_rope_quarter_turn() -> None:
    """Verify a quarter-turn rotary embedding with known values.

    At 90 degrees, cos = 0 and sin = 1, so the result is rotate_half(x).
    Q = [1, 2, 3, 4] becomes [-3, -4, 1, 2] and K = [5, 6, 7, 8]
    becomes [-7, -8, 5, 6].
    """
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
    """Verify broadcasting when the coefficients are unsqueezed at axis 2.

    The test uses head-specific coefficients: head 0 rotates by 90
    degrees and head 1 stays unchanged at both token positions.
    Since K = 2 * Q and rotation is linear, its output is twice Q's.
    """
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
    """Verify that disabled CPU autocast remains disabled in a no-op context.

    With the outer context disabled and enabled=False, entering
    maybe_autocast must not enable mixed precision.
    """
    # Start with autocast disabled
    with torch.autocast(device_type="cpu", enabled=False):
        with maybe_autocast(
            "cpu", dtype=torch.bfloat16, enabled=False
        ):
            assert not torch.is_autocast_enabled("cpu")

def test_maybe_autocast_enabled() -> None:
    """Verify that CPU autocast is enabled only within the inner context.

    Starting with autocast disabled, enabled=True should activate it
    inside the context and restore the disabled state on exit.
    """
    # Autocast should turn on temporarily
    with torch.autocast(device_type="cpu", enabled=False):
        with maybe_autocast(
            "cpu", dtype=torch.bfloat16, enabled=True
        ):
            assert torch.is_autocast_enabled("cpu")

        # Leaving the context restores the original state
        assert not torch.is_autocast_enabled("cpu")

def test_maybe_autocast_nested_disabled() -> None:
    """Verify temporary disabling of an active CPU autocast context.

    Starting with autocast enabled, enabled=False must disable it
    inside the nested context and restore the enabled state on exit.
    """
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


