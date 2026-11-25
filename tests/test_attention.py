
import pytest
import torch
from slm.functional import attention
from torch.nn.functional import scaled_dot_product_attention


def test_attention_parity():
    """Match PyTorch's scaled dot-product attention output.

    Use the same random tensor for Q, K and V in both implementations.
    With the default scale and no mask or dropout, their outputs should
    agree within floating-point tolerance.
    """
    x = torch.randn((3, 2, 3))
    sdpa_val = scaled_dot_product_attention(x, x, x)
    custom_attention_score, _= attention(x, x, x)

    torch.testing.assert_close(sdpa_val, custom_attention_score), "Attention implementation differs!"

# Additional attention tests
def test_attention_uniform_weights() -> None:
    """Verify uniform attention when all query-key scores are equal.

    Zero Q and K give scores [0, 0], so softmax gives [0.5, 0.5]
    for each query. With V = [[2, 4], [6, 8]], the weighted sum is
    0.5 * [2, 4] + 0.5 * [6, 8] = [4, 6] for both queries.
    """
    # All attention scores are equal
    q = torch.zeros(1, 2, 2)
    k = torch.zeros(1, 2, 2)
    v = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    result, weights = attention(q, k, v, dropout=0.0)

    expected_weights = torch.full((1, 2, 2), 0.5)
    expected_result = torch.tensor([
        [[4.0, 6.0], [4.0, 6.0]]
    ])

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(result, expected_result)


def test_attention_mask_blocks_keys() -> None:
    """Verify that masked keys contribute no attention weight.

    Both queries can attend only to the first key, so their weights
    must be [1, 0]. The output is 1 * [2, 4] + 0 * [6, 8]
    = [2, 4] for each query.
    """
    # Both queries can attend only to the first key
    q = torch.zeros(1, 2, 2)
    k = torch.zeros(1, 2, 2)
    v = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    mask = torch.tensor([
        [[True, False], [True, False]]
    ])

    result, weights = attention(
        q, k, v, mask=mask, dropout=0.0
    )

    expected_weights = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]]
    ])
    expected_result = torch.tensor([
        [[2.0, 4.0], [2.0, 4.0]]
    ])

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(result, expected_result)


def test_attention_custom_scale_parity() -> None:
    """Match PyTorch attention with a custom scale and boolean mask.

    Use two queries, three keys and scale 0.7 in both implementations.
    With dropout disabled, the outputs should agree, every weight row
    should sum to one, and all masked positions should have zero weight.
    """
    # Compare with the official PyTorch implementation
    q = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]]
    ])
    k = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    ])
    v = torch.tensor([
        [[2.0, 0.0], [0.0, 4.0], [6.0, 8.0]]
    ])

    mask = torch.tensor([
        [[True, False, True], [False, True, True]]
    ])

    result, weights = attention(
        q, k, v,
        mask=mask,
        scale=0.7,
        dropout=0.0,
    )

    expected = scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        scale=0.7,
        dropout_p=0.0,
    )

    torch.testing.assert_close(
        result, expected, rtol=1e-5, atol=1e-6
    )

    # Attention weights should sum to 1
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(1, 2),
    )

    # Masked-out positions should receive zero weight
    assert torch.all(weights[~mask] == 0)

def test_attention_dropout_one() -> None:
    """Verify that full dropout removes every attention weight.

    With dropout probability 1, all weights are zero, so their weighted
    sum of the value vectors must also be zero.
    """
    # 100% dropout should zero all attention weights
    q = torch.zeros(1, 2, 2)
    k = torch.zeros(1, 2, 2)
    v = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    result, weights = attention(
        q, k, v, dropout=1.0
    )

    torch.testing.assert_close(
        weights, torch.zeros_like(weights)
    )
    torch.testing.assert_close(
        result, torch.zeros_like(result)
    )


def test_attention_backward() -> None:
    """Verify that attention supports backpropagation through Q, K and V.

    Use the sum of squared outputs as a scalar loss. Backpropagation
    should populate finite gradients for each of the three input tensors.
    This checks gradient connectivity, not exact gradient values.
    """
    # Check whether gradients propagate through attention
    q = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]]
    ], requires_grad=True)

    k = torch.tensor([
        [[1.0, 1.0], [1.0, 0.0]]
    ], requires_grad=True)

    v = torch.tensor([
        [[2.0, 4.0], [6.0, 8.0]]
    ], requires_grad=True)

    result, _ = attention(q, k, v, dropout=0.0)
    loss = result.square().sum()
    loss.backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
