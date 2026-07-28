"""aieng — the library you build across the AI Engineering Roadmap.

Each subpackage corresponds to a phase of the roadmap. Nothing here is a
wrapper around a framework; it is the from-scratch implementation you write
while working through the books, kept honest by the tests in ``tests/``.

    aieng.classic      Phase 1-2   Geron        pipelines, model selection
    aieng.nn           Phase 2     Geron        training loops, schedules
    aieng.tokenizer    Phase 5     Raschka 2    byte pair encoding
    aieng.transformer  Phase 5     Raschka 3-5  attention, GPT, sampling
    aieng.finetune     Phase 6     Raschka 6-7  SFT, LoRA, classification heads
    aieng.rag          Phase 4,7   Alammar 8    chunking, retrieval, fusion
    aieng.evals        Phase 7     Huyen 3-4    metrics, judges, harness
    aieng.serving      Phase 7     Huyen 9      batching, memory arithmetic
    aieng.agents       Phase 8     Lanham       tools, memory, the loop

Heavy dependencies (torch, transformers) are imported lazily inside the
subpackages that need them, so ``import aieng`` stays fast and the modules
that do not need torch remain usable without it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "agents",
    "classic",
    "evals",
    "finetune",
    "nn",
    "rag",
    "serving",
    "tokenizer",
    "transformer",
]


def __getattr__(name: str):
    """Import subpackages lazily so torch is only loaded when actually needed."""
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
