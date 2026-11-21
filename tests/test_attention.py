
import pytest
import torch
from slm.functional import attention
from torch.nn.functional import scaled_dot_product_attention


def test_attention_parity() -> None:
    x = torch.randn((3, 2, 3))
    sdpa_val = scaled_dot_product_attention(x, x, x)
    custom_attention_score, _= attention(x, x, x)

    torch.testing.assert_close(sdpa_val, custom_attention_score), "Attention implementation differs!"
