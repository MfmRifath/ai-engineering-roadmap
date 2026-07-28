"""Train/test splitting that survives a growing dataset — Geron ch. 2.

A seeded ``train_test_split`` is stable only while the dataset is. Add rows and
everything reshuffles, so instances that were in the test set move into training
— which silently invalidates every result you have ever reported on that data.

Hashing a stable identifier fixes this: each row's side of the split is a
function of its own id, so it never changes no matter what else arrives.
"""

from __future__ import annotations

from collections.abc import Sequence
from zlib import crc32


def _in_test_set(identifier: int | str, test_ratio: float) -> bool:
    """Deterministic per-row assignment via CRC32 of the identifier."""
    key = str(identifier).encode("utf-8")
    return crc32(key) & 0xFFFFFFFF < test_ratio * 2**32


def hash_split(
    identifiers: Sequence[int | str], test_ratio: float = 0.2
) -> tuple[list[int], list[int]]:
    """Split by hashed identifier. Stable as the dataset grows.

    Returns ``(train_indices, test_indices)`` — positions in ``identifiers``, so
    it works with any container.

    The guarantee worth stating: appending rows never moves an existing row
    between splits. Verified in ``tests/test_splits.py``.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    train, test = [], []
    for i, ident in enumerate(identifiers):
        (test if _in_test_set(ident, test_ratio) else train).append(i)
    return train, test


def stratified_bins(values: Sequence[float], bins: Sequence[float]) -> list[int]:
    """Assign each value to a bin, for stratified sampling — Geron ch. 2.

    Purely random splits are fine on large datasets and dangerous on small or
    skewed ones: if an important variable is unevenly represented in the test
    set, the test score measures the wrong population. Binning the variable and
    sampling proportionally within bins fixes it.

    ``bins`` are upper edges; values above the last edge land in the final bin.
    """
    if not bins:
        raise ValueError("bins must not be empty")

    out: list[int] = []
    for v in values:
        assigned = len(bins) - 1
        for i, edge in enumerate(bins):
            if v <= edge:
                assigned = i
                break
        out.append(assigned)
    return out
