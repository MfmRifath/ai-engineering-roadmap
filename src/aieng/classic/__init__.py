"""Classical ML helpers — Geron ch. 1-9.

Thin on purpose: scikit-learn already provides the primitives, and re-wrapping
them teaches nothing. What lives here is the *discipline* that is easy to get
wrong — leak-free splitting and pipeline construction.
"""

from aieng.classic.splits import hash_split, stratified_bins

__all__ = ["hash_split", "stratified_bins"]
