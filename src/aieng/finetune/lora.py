"""LoRA from scratch — Raschka Appendix E, Alammar ch. 12, Huyen ch. 7.

The observation: the *update* to a weight matrix during fine-tuning has low
intrinsic rank. You are not teaching the model language again, you are nudging
it. So freeze W and learn a low-rank decomposition of the change::

    W' = W + BA        B: (d, r)   A: (r, k)   r << min(d, k)

For a 4096x4096 matrix at r=8 that is 65k trainable parameters instead of 16.7M
— about 0.4%. Gradients and optimizer states exist only for A and B, which is
why the 56 GB of Adam state in a 7B full fine-tune collapses to megabytes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """The low-rank update ``x @ A @ B * (alpha / rank)``.

    ``A`` is initialized with Kaiming-uniform noise and ``B`` with **zeros**, so
    ``BA == 0`` at initialization and the adapted model starts out behaving
    exactly like the base model. If both were zero nothing would ever learn —
    A's randomness is what breaks the symmetry.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # zero: start at identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A @ self.B) * self.scaling


class LinearWithLoRA(nn.Module):
    """Wrap an ``nn.Linear`` so its output gains a trainable low-rank update.

    The base layer is frozen; only the adapter trains.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False  # freeze the base

        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weights — zero inference overhead.

        Merge into an **fp16/fp32** base, never a quantized one, or you compound
        quantization error.
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        delta = (self.lora.A @ self.lora.B).T * self.lora.scaling
        merged.weight.copy_(self.linear.weight + delta)
        if self.linear.bias is not None:
            merged.bias.copy_(self.linear.bias)
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = ("W_query", "W_key", "W_value"),
) -> nn.Module:
    """Replace the named ``nn.Linear`` submodules with LoRA-wrapped versions.

    Attention projections are the standard target; including the feedforward
    layers helps more and costs more. Everything not targeted is frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LinearWithLoRA(child, rank, alpha))
    return model


def count_trainable(model: nn.Module) -> tuple[int, int, float]:
    """Return ``(trainable, total, percentage)``.

    A healthy LoRA setup reports something like 0.1-1%. If it reports 100%, you
    forgot to freeze the base.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100.0 * trainable / max(total, 1)
