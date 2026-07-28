"""The test that proves Raschka chapter 3 is genuinely done.

If ``MultiHeadAttention`` matches ``torch.nn.MultiheadAttention`` to
floating-point tolerance with transferred weights, then the projections, the
head splitting, the scaling, and the causal mask are all correct. That converts
"I think I built it right" into proof.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="attention tests require PyTorch")
import torch.nn as nn  # noqa: E402

from aieng.transformer.attention import (  # noqa: E402
    CausalAttention,
    MultiHeadAttention,
    SelfAttention,
    SelfAttentionV1,
)

pytestmark = pytest.mark.torch


@pytest.fixture
def seed():
    torch.manual_seed(123)


# --------------------------------------------------------------------------
# The headline test
# --------------------------------------------------------------------------


def test_matches_pytorch_multihead_attention(seed):
    """Ours must equal torch's, given the same weights."""
    batch, seq_len, d_model, n_heads = 2, 6, 32, 4

    mine = MultiHeadAttention(
        d_in=d_model,
        d_out=d_model,
        context_length=seq_len,
        dropout=0.0,
        num_heads=n_heads,
        qkv_bias=False,
    ).eval()

    theirs = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=n_heads, dropout=0.0, bias=False, batch_first=True
    ).eval()

    with torch.no_grad():
        # torch packs Q, K, V into one (3E, E) matrix in that order. nn.Linear
        # stores weight as (out, in) and computes x @ W.T, the same convention.
        theirs.in_proj_weight.copy_(
            torch.cat([mine.W_query.weight, mine.W_key.weight, mine.W_value.weight], dim=0)
        )
        theirs.out_proj.weight.copy_(mine.out_proj.weight)
        # torch's out_proj has no bias when bias=False, so zero ours to match.
        mine.out_proj.bias.zero_()

    x = torch.randn(batch, seq_len, d_model)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    with torch.no_grad():
        ours = mine(x)
        expected, _ = theirs(x, x, x, attn_mask=causal, need_weights=False)

    torch.testing.assert_close(ours, expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# Properties that must hold regardless of weights
# --------------------------------------------------------------------------


def test_causal_mask_zeroes_the_future(seed):
    """No position may attend to a later one."""
    seq_len = 8
    mha = MultiHeadAttention(16, 16, seq_len, dropout=0.0, num_heads=4).eval()

    with torch.no_grad():
        weights = mha.attention_weights(torch.randn(1, seq_len, 16))

    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[..., upper] == 0.0), "attended to a future position"


def test_attention_weights_sum_to_one(seed):
    """Masking must happen before the softmax, so rows stay normalized.

    This is the test that catches the classic bug of zeroing weights *after*
    the softmax, which leaves them un-normalized.
    """
    seq_len = 8
    mha = MultiHeadAttention(16, 16, seq_len, dropout=0.0, num_heads=4).eval()

    with torch.no_grad():
        weights = mha.attention_weights(torch.randn(2, seq_len, 16))

    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), rtol=1e-5, atol=1e-5
    )


def test_changing_a_future_token_cannot_change_an_earlier_output(seed):
    """The strongest statement of causality, tested behaviourally."""
    seq_len = 6
    mha = MultiHeadAttention(16, 16, seq_len, dropout=0.0, num_heads=2).eval()

    x = torch.randn(1, seq_len, 16)
    x_modified = x.clone()
    x_modified[0, -1] = torch.randn(16)  # change only the LAST token

    with torch.no_grad():
        out, out_modified = mha(x), mha(x_modified)

    torch.testing.assert_close(out[0, :-1], out_modified[0, :-1], rtol=1e-6, atol=1e-6)


def test_output_shape_is_preserved(seed):
    """Shape preservation is what lets blocks be stacked."""
    mha = MultiHeadAttention(32, 32, 16, dropout=0.0, num_heads=8).eval()
    x = torch.randn(4, 10, 32)
    assert mha(x).shape == x.shape


def test_handles_sequences_shorter_than_context_length(seed):
    """The mask buffer is context_length square and must be sliced."""
    mha = MultiHeadAttention(16, 16, context_length=64, dropout=0.0, num_heads=4).eval()
    for seq_len in (1, 5, 63, 64):
        assert mha(torch.randn(1, seq_len, 16)).shape == (1, seq_len, 16)


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_in=16, d_out=30, context_length=8, num_heads=4)


def test_head_dim_splits_rather_than_duplicates():
    """Multi-head must not multiply the parameter count."""
    single = MultiHeadAttention(64, 64, 8, num_heads=1)
    multi = MultiHeadAttention(64, 64, 8, num_heads=8)
    assert multi.head_dim == 8
    assert sum(p.numel() for p in single.parameters()) == sum(p.numel() for p in multi.parameters())


# --------------------------------------------------------------------------
# The earlier versions in the progression
# --------------------------------------------------------------------------


def test_v1_is_a_weighted_average(seed):
    """With no trainable weights, attention is literally a weighted average."""
    x = torch.randn(1, 5, 8)
    out = SelfAttentionV1()(x)
    assert out.shape == x.shape
    # Every output lies within the convex hull of the inputs.
    assert out.max() <= x.max() + 1e-5
    assert out.min() >= x.min() - 1e-5


def test_v1_has_no_parameters():
    assert sum(p.numel() for p in SelfAttentionV1().parameters()) == 0


def test_self_attention_is_permutation_equivariant(seed):
    """Without a mask or positions, attention sees a *set*.

    This is exactly why positional encodings are required — the model literally
    cannot distinguish "dog bites man" from "man bites dog".
    """
    attn = SelfAttention(8, 8).eval()
    x = torch.randn(1, 4, 8)
    perm = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        permuted_then_attended = attn(x[:, perm])
        attended_then_permuted = attn(x)[:, perm]

    torch.testing.assert_close(permuted_then_attended, attended_then_permuted, rtol=1e-5, atol=1e-5)


def test_causal_attention_matches_multihead_with_one_head(seed):
    """CausalAttention is MultiHeadAttention with num_heads=1, minus out_proj."""
    seq_len, d = 6, 16
    causal = CausalAttention(d, d, seq_len, dropout=0.0).eval()
    mha = MultiHeadAttention(d, d, seq_len, dropout=0.0, num_heads=1).eval()

    with torch.no_grad():
        mha.W_query.weight.copy_(causal.W_query.weight)
        mha.W_key.weight.copy_(causal.W_key.weight)
        mha.W_value.weight.copy_(causal.W_value.weight)
        mha.out_proj.weight.copy_(torch.eye(d))  # make out_proj a no-op
        mha.out_proj.bias.zero_()

        x = torch.randn(1, seq_len, d)
        torch.testing.assert_close(causal(x), mha(x), rtol=1e-5, atol=1e-5)


def test_dropout_is_inactive_in_eval_mode(seed):
    """A generation-time bug: forgetting model.eval() degrades output randomly."""
    mha = MultiHeadAttention(16, 16, 8, dropout=0.5, num_heads=4)
    x = torch.randn(1, 8, 16)

    mha.eval()
    with torch.no_grad():
        torch.testing.assert_close(mha(x), mha(x), rtol=1e-6, atol=1e-6)

    mha.train()
    with torch.no_grad():
        assert not torch.allclose(mha(x), mha(x))
