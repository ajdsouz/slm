import math
from typing import Literal
import torch
from jaxtyping import jaxtyped, Float, Shaped
from beartype import beartype as typechecker
from torch import Tensor 
import torch.nn.functional as F 
from contextlib import nullcontext

def attention(
        query: Float[Tensor, "batch query_sequence d_head"]
            | Float[Tensor, "batch n_heads query_sequence d_head"],
        key: Float[Tensor, "batch key_sequence d_head"]
            | Float[Tensor, "batch n_heads key_sequence d_head"],
        value: Float[Tensor, "batch key_sequence d_value"]
            | Float[Tensor, "batch n_heads key_sequence d_value"],
        scale: float | None = None, 
        dropout: float = 0.0,
        mask: Shaped[Tensor, "#query_sequence #key_sequence"]
            | Shaped[Tensor, "#batch #query_sequence #key_sequence"]
            | Shaped[Tensor, "#batch #n_heads #query_sequence #key_sequence"]
            | None = None
) -> tuple[
    Float[Tensor, "batch query_sequence d_value"]
        | Float[Tensor, "batch n_heads query_sequence d_value"],
    Float[Tensor, "batch query_sequence key_sequence"]
        | Float[Tensor, "batch n_heads query_sequence key_sequence"],
]:
    """Compute attention for batched single-head or multi-head inputs.

    Args:
        query: Queries shaped [batch, query_sequence, d_head] or
            [batch, n_heads, query_sequence, d_head].
        key: Keys with matching rank, batch, heads and feature dimension.
            Their sequence length may differ from query.
        value: Values with the same rank, batch, heads and sequence length as
            key. Their feature dimension may differ from query and key.
        scale: Score multiplier; defaults to the inverse square root of d_head.
        dropout: Dropout probability for the attention weights.
        mask: A 2D [query_sequence, key_sequence] mask or a mask with the same
            rank as query. Each mask axis may have size one for broadcasting.
            Zero blocks attention. For multi-head inputs, use a 4D mask to
            make batch and head axes explicit.

    Returns:
        Weighted values and attention weights after dropout, with the same
        rank, batch size and number of heads as query. Query, key and value
        batch/head axes must match; only the mask is broadcastable.
    """

    if not query.ndim == key.ndim == value.ndim:
        raise ValueError('query, key and value must have the same rank')
    if mask is not None and mask.ndim not in (2, query.ndim):
        raise ValueError('mask must be 2D or have the same rank as query')

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
    x: Float[Tensor, "d_head"]
        | Float[Tensor, "sequence d_head"]
        | Float[Tensor, "batch sequence d_head"]
        | Float[Tensor, "batch n_heads sequence d_head"],
) -> (
    Float[Tensor, "d_head"]
    | Float[Tensor, "sequence d_head"]
    | Float[Tensor, "batch sequence d_head"]
    | Float[Tensor, "batch n_heads sequence d_head"]
):
    """Rotate paired halves of the last dimension.

    Args:
        x: One head vector, a sequence of vectors, a batched sequence, or a
            multi-head tensor [batch, n_heads, sequence, d_head]. The last
            dimension contains two equal-sized halves.

    Returns:
        Rotated tensor with the same shape as x.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
        q: Float[Tensor, "batch query_heads sequence d_head"],
        k: Float[Tensor, "batch kv_heads sequence d_head"],
        cos: Float[Tensor, "#batch #{q.shape[3-unsqueeze_dim]} d_head"],
        sin: Float[Tensor, "#batch #{q.shape[3-unsqueeze_dim]} d_head"],
        unsqueeze_dim: Literal[1, 2] = 1
) -> tuple[
    Float[Tensor, "batch query_heads sequence d_head"],
    Float[Tensor, "batch kv_heads sequence d_head"],
]:
    """Apply RoPE

    Args:
        q: Query tensor [batch, query_heads, sequence, d_head].
        k: Key tensor [batch, kv_heads, sequence, d_head]. Its head count may
            differ from query for grouped-query attention.
        cos: With unsqueeze_dim=1, cosine values [batch, sequence, d_head];
            batch and sequence may have size one. With unsqueeze_dim=2,
            values are [batch, heads, d_head] and shared across sequence.
        sin: Sine values with the same layout rules as cos.
        unsqueeze_dim: Insert the head axis (1, the default), or the sequence
            axis (2). Per-head rotations require equal query/key head counts
            unless the rotation's head axis has size one.

    Returns:
        Rotated query and key with their original shapes and head counts.
    """
    if unsqueeze_dim == 2:
        for rotation in (cos, sin):
            if rotation.shape[1] not in (1, k.shape[1]):
                raise ValueError('Per-head rotations must match both query and key heads')
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
 
 
 
 
 
