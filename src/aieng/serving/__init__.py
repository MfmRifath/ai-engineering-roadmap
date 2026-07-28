"""Serving arithmetic — Huyen ch. 7 and 9.

The calculators here turn "will this fit?" and "how fast can this possibly be?"
from guesswork into arithmetic you can do before renting a GPU.
"""

from aieng.serving.budget import (
    MemoryBudget,
    decode_floor_ms,
    inference_memory_gb,
    kv_cache_gb,
    training_memory_gb,
)

__all__ = [
    "MemoryBudget",
    "decode_floor_ms",
    "inference_memory_gb",
    "kv_cache_gb",
    "training_memory_gb",
]
