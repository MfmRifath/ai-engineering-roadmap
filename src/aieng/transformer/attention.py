"""Attention, built four times — Raschka ch. 3.

The progression is the explanation, so all four versions are kept rather than
only the final one:

    SelfAttentionV1      no trainable weights   — attention is a weighted average
    SelfAttention        trainable Q, K, V      — the model learns the weights
    CausalAttention      + masking and dropout  — it cannot see the future
    MultiHeadAttention   + parallel heads       — it can attend to several things

Correctness check: ``MultiHeadAttention`` must match
``torch.nn.MultiheadAttention`` to floating-point tolerance. See
``tests/test_attention.py`` — if that passes, chapter 3 is genuinely done.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    """Simplified self-attention with **no trainable parameters**.

    Included because it is the clearest possible statement of what attention is:
    score every pair by dot product, softmax the scores into weights that sum to
    one, and take the weighted average of the inputs.

    Its limitation is the reason the next version exists — nothing is learned, so
    relevance is fixed by whatever the input vectors happen to be.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, T, C)
        attn_scores = x @ x.transpose(-2, -1)  # dot product as similarity
        attn_weights = torch.softmax(attn_scores, dim=-1)  # rows sum to 1
        return attn_weights @ x  # weighted average


class SelfAttention(nn.Module):
    """Self-attention with trainable Q, K, V projections.

    Three separate projections because *what makes a token findable* (key) is not
    the same as *what it contributes* (value), and neither is the same as *what a
    position is looking for* (query). Collapsing them would force the model to
    conflate the three.

    ``nn.Linear(bias=False)`` rather than ``nn.Parameter(torch.rand(...))``:
    mathematically identical, but the initialization is scaled to fan-in instead
    of uniform [0, 1), and it trains noticeably better.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)
        attn_scores = queries @ keys.transpose(-2, -1)
        # / sqrt(d_k): dot products of d-dimensional vectors have variance ~d,
        # and large scores saturate the softmax where gradients vanish.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class CausalAttention(nn.Module):
    """Self-attention that cannot look ahead.

    The mask is applied to the **scores** with ``-inf`` *before* the softmax, not
    to the weights after it. ``exp(-inf) == 0``, so masked positions get exactly
    zero weight and the softmax still normalizes over the visible positions.
    Zeroing weights after the softmax leaves them un-normalized, which is the
    classic implementation bug.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # A buffer, not a parameter: it moves with the model and is saved in the
        # state dict, but it receives no gradient. A mask is structure, not
        # something to learn.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_tokens, _ = x.shape
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        # Slice the buffer: it is context_length square, the input may be shorter.
        attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return self.dropout(attn_weights) @ values


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention — the version that goes in the model.

    One attention operation produces exactly one weighted average, so it can
    attend to exactly one thing. Multiple heads attend in different learned
    subspaces at once — one may track syntactic agreement while another tracks
    coreference.

    ``head_dim = d_out // num_heads``: the output dimension is **split** among
    heads, not duplicated, so total parameters match single-head attention. Head
    specialization comes for free.

    The implementation projects once and reshapes rather than running
    ``num_heads`` separate modules — identical mathematics, one large GEMM
    instead of many small ones, which is dramatically better on a GPU.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 1,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by num_heads ({num_heads})")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Mixes information across heads after concatenation. Not decorative —
        # it is in the original architecture and omitting it costs quality.
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        # (B, T, d_out) -> (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # (B, heads, T, T)
        attn_scores = attn_scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        # Scale by head_dim, not d_out — the dot products are per head.
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)  # (B, T, heads, head_dim)
        # .contiguous() is required: transpose returns a non-contiguous view and
        # .view() cannot reinterpret it.
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention weight matrix, ``(B, heads, T, T)``, for inspection.

        Useful for plotting per-head heatmaps. Note the standard caveat: these
        show where the model *looked*, not why it decided.
        """
        b, num_tokens, _ = x.shape
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)

        scores = queries @ keys.transpose(2, 3)
        scores = scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        return torch.softmax(scores / self.head_dim**0.5, dim=-1)
