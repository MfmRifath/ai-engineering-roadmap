"""Fine-tuning — Raschka ch. 6-7 and Appendix E, Alammar ch. 11-12, Huyen ch. 7.

LoRA is implemented from scratch (about 30 lines) because doing it by hand makes
the memory arithmetic in ``aieng.serving.budget`` concrete: freezing W and
learning a low-rank update BA is what makes the optimizer state disappear.
"""

from __future__ import annotations

__all__ = ["LinearWithLoRA", "LoRALayer", "apply_lora", "mask_prompt_tokens"]


def __getattr__(name: str):
    import importlib

    if name in {"LoRALayer", "LinearWithLoRA", "apply_lora"}:
        return getattr(importlib.import_module(f"{__name__}.lora"), name)
    if name == "mask_prompt_tokens":
        return getattr(importlib.import_module(f"{__name__}.collate"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
