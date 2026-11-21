import math
import torch
from jaxtyping import jaxtyped, Float, Shaped
from beartype import beartype as typechecker
from torch import Tensor 
import torch.nn.functional as F 
from contextlib import nullcontext

def attention(
        query: Float[Tensor, "*#batch query_sequence d_head"],
        key: Float[Tensor, "*#batch key_sequence d_head"],
        value: Float[Tensor, "*#batch key_sequence d_value"],
        scale: float | None = None, 
        dropout: float = 0.0,
        mask: Shaped[Tensor, "..."] | None = None
) -> tuple[
    Float[Tensor, "*#batch query_sequence d_value"],
    Float[Tensor, "*#batch query_sequence key_sequence"],
]:
    """Compute scaled dot-product attention over the last two dimensions.

    Args:
        query: Floating-point queries with optional batch and head dimensions.
        key: Keys with the same feature dimension as query.
        value: Values with the same sequence length as key.
        scale: Score multiplier; defaults to the inverse square root of d_head.
        dropout: Dropout probability for the attention weights.
        mask: Mask broadcastable to the attention scores; zero blocks attention.

    Returns:
        The weighted values and attention weights after dropout. Leading
        dimensions follow PyTorch broadcasting rules.
    """

    if scale is None:
        scale = 1 / math.sqrt(key.size(-1))

    scores = (query @ key.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout)
    attention = attention_weights @ value 

    return attention, attention_weights


def split_heads(
    x: Float[Tensor, "batch sequence d_model"], 
    n_heads: int
) -> Float[Tensor, "batch {n_heads} sequence d_model//{n_heads}"]:
    """Split tensor into n_heads tensors for individual attention heads

    Args:
        x (Tensor): tensor to be split
        n_heads (int): number of heads

    Returns:
        Tensor: A different view of the tensor of form [Batch n_heads Sequence d_head]
    """
    # [batch sequence d_model]
    B, S, D = x.shape
    d_head = D // n_heads
    # [batch sequence n_heads d_head] => [batch n_heads sequence d_head]
    return x.view(B, S, n_heads, d_head).transpose(1, 2)

def merge_heads(
    x: Float[Tensor, "batch n_heads sequence d_head"]
) -> Float[Tensor, "batch sequence n_heads*d_head"]:
    """Merge tensor after after attention calculation

    Args:
        x (Tensor): Tensor [Batch n_heads Sequence d_head] to be merged

    Returns:
        Tensor: Merged tensor of shape [Batch Sequence d_model]
    """
    B, Nh, S, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, Nh * Dh)
 
def rotate_half(
    x: Float[Tensor, "*batch d_head"],
) -> Float[Tensor, "*batch d_head"]:
    """ Rotate the tensor by 90° on last dim

    Args:
        x (Tensor): tensor to be rotated

    Returns:
        Tensor: rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
        q: Float[Tensor, "*#q_dims d_head"],
        k: Float[Tensor, "*#k_dims d_head"],
        cos: Float[Tensor, "... #d_head"],
        sin: Float[Tensor, "... #d_head"],
        unsqueeze_dim: int = 1
) -> tuple[
    Float[Tensor, "*#q_dims d_head"],
    Float[Tensor, "*#k_dims d_head"],
]:
    """Apply RoPE

    Args:
        q (Tensor): Query tensor, commonly [batch, query_heads, sequence, d_head].
        k (Tensor): Key tensor; its number of heads may differ from q.
        cos (Tensor): Cosine values, broadcastable to q and k after unsqueezing.
        sin (Tensor): Sine values, broadcastable to q and k after unsqueezing.
        unsqueeze_dim (int, optional): unsqueeze sine and cosine tensors across specific dimension. Defaults to 1.

    Returns:
        tuple[Tensor, Tensor]: Rotated query and key tensors, with leading
            dimensions determined by broadcasting. Their head counts stay
            independent, including when unsqueeze_dim selects another layout.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
 
def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
) -> torch.autocast | nullcontext[None]:
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()
 
 
def repeat_kv(
    hidden_states: Shaped[Tensor, "batch n_kv_heads sequence d_head"],
    n_rep: int,
) -> Shaped[Tensor, "batch n_kv_heads*{n_rep} sequence d_head"]:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
 
 
 
 
 
