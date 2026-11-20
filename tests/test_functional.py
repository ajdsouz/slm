import pytest
import torch

from slm.functional import (
    rotate_half,
    split_heads,
    merge_heads,
    repeat_kv,
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

