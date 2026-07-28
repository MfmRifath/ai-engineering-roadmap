"""The transformer, from scratch — Raschka ch. 3-5, Alammar ch. 3.

Imports are lazy because these modules need PyTorch, and the rest of ``aieng``
should stay usable without it.

    from aieng.transformer import MultiHeadAttention, GPTModel, generate
"""

from __future__ import annotations

__all__ = [
    "GPT_CONFIG_124M",
    "CausalAttention",
    "FeedForward",
    "GPTModel",
    "LayerNorm",
    "MultiHeadAttention",
    "SelfAttention",
    "TransformerBlock",
    "generate",
    "generate_simple",
]


def __getattr__(name: str):
    import importlib

    for module in ("attention", "gpt", "sampling"):
        mod = importlib.import_module(f"{__name__}.{module}")
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
