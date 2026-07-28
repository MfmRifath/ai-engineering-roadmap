"""Retrieval — Alammar ch. 8, Huyen ch. 6.

Deliberately dependency-free at the core: chunking and rank fusion are pure
Python, so they are testable without downloading an embedding model. The parts
that need ``sentence-transformers`` are imported lazily.
"""

from aieng.rag.chunking import Chunk, chunk_by_structure, chunk_text
from aieng.rag.fusion import maximal_marginal_relevance, reciprocal_rank_fusion

__all__ = [
    "Chunk",
    "chunk_by_structure",
    "chunk_text",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
]
