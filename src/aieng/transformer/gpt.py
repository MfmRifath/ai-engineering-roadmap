"""A GPT model, from scratch — Raschka ch. 4.

Embeddings, a stack of identical blocks, a linear head. The block does exactly
two things: attention moves information *between* positions, the feedforward
network processes *each position independently*. Both are wrapped in
pre-normalized residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aieng.transformer.attention import MultiHeadAttention

# GPT-2 small. Learn these numbers; every larger variant scales the same dials.
GPT_CONFIG_124M: dict[str, int | float | bool] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,  # head_dim = 768 / 12 = 64
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT_CONFIGS: dict[str, dict] = {
    "gpt2-small": {**GPT_CONFIG_124M},
    "gpt2-medium": {**GPT_CONFIG_124M, "emb_dim": 1024, "n_heads": 16, "n_layers": 24},
    "gpt2-large": {**GPT_CONFIG_124M, "emb_dim": 1280, "n_heads": 20, "n_layers": 36},
    "gpt2-xl": {**GPT_CONFIG_124M, "emb_dim": 1600, "n_heads": 25, "n_layers": 48},
}


class LayerNorm(nn.Module):
    """Layer normalization, implemented rather than imported.

    Normalizes across the **feature** dimension within a single example, so it is
    independent of batch size and composition — identical at training and
    inference, and it works at batch size 1, which generation requires.

    ``unbiased=False`` (divide by n, not n-1) matches GPT-2's original
    TensorFlow implementation. Using the unbiased estimator makes loaded OpenAI
    weights produce subtly wrong outputs.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        # scale and shift are learned, so the network can undo the
        # normalization if it needs to — no expressiveness is lost.
        return self.scale * norm + self.shift


class GELU(nn.Module):
    """The tanh approximation of GELU, as used by GPT-2.

    Smooth, and unlike ReLU it is nonzero for small negative inputs — so no dead
    neurons and a smoother optimization landscape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Position-wise feedforward network: expand 4x, activate, contract.

    Holds roughly two-thirds of a transformer's non-embedding parameters
    (768->3072->768 is ~4.7M per block vs ~2.4M for attention). Evidence
    suggests it acts as a key-value memory for factual associations, which is
    why knowledge-editing research targets it.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        emb = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            GELU(),
            nn.Linear(4 * emb, emb),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """One transformer block. Pre-norm, two residual sublayers.

    ``x = x + attention(norm(x))`` then ``x = x + ffn(norm(x))``.

    Pre-norm (normalize *before* the sublayer) rather than the original paper's
    post-norm: it trains far more stably at depth because the residual path
    stays clean, so gradients reach the embeddings unimpeded.

    The block **preserves its input shape**, which is what allows stacking.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.drop_shortcut(self.att(self.norm1(x)))
        x = x + shortcut  # residual: gradients get a direct path backwards

        shortcut = x
        x = self.drop_shortcut(self.ff(self.norm2(x)))
        return x + shortcut


class GPTModel(nn.Module):
    """A decoder-only transformer.

    ``tokens -> embeddings + positions -> N blocks -> norm -> LM head -> logits``

    Note on parameter count: the naive sum for GPT-2 small is ~163M, but the
    published model is 124M. The difference is **weight tying** — the output
    head shares the token embedding matrix (50257 x 768 = 38.6M). Pass
    ``tie_weights=True`` to reproduce that.
    """

    def __init__(self, cfg: dict, tie_weights: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if tie_weights:
            # "which token is this" and "which token comes next" share a geometry
            self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length "
                f"{self.cfg['context_length']} — there is no positional embedding "
                f"for position {seq_len - 1}"
            )

        tok = self.tok_emb(in_idx)
        pos = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok + pos)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))

    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.pos_emb.weight.numel()
            if self.out_head.weight is not self.tok_emb.weight:
                total -= self.tok_emb.weight.numel()
        return total

    def memory_mb(self, bytes_per_param: int = 4) -> float:
        """Weight memory only — see ``aieng.serving`` for the full training budget."""
        return self.num_params() * bytes_per_param / 1e6
