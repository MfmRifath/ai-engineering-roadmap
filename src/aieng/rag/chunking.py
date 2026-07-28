"""Chunking — where most RAG quality is won or lost.

Alammar ch. 8 and Huyen ch. 6 both make the point that chunking is the highest-
leverage hyperparameter in a RAG system, and that it interacts with document
type. This module makes it a *measurable* choice rather than a default.

Two invariants the tests enforce:

* every chunk is within the size bound;
* consecutive chunks overlap by the requested amount, so an answer spanning a
  boundary is not split into two unretrievable halves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A retrievable unit of text, with enough metadata to cite it."""

    text: str
    start: int
    end: int
    index: int
    source: str | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def with_context(self, document_summary: str) -> str:
        """Contextual retrieval (Huyen ch. 6).

        Prepending a document-level summary means an isolated chunk carries
        enough context to be findable on its own. Cheap to implement and
        usually a large recall gain.
        """
        return f"[Document context: {document_summary}]\n\n{self.text}"


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source: str | None = None,
) -> list[Chunk]:
    """Fixed-size chunking with overlap, snapped to word boundaries.

    Sizes are in characters. Roughly 4 characters per token for English, so
    ``chunk_size=512`` is ~128 tokens; measure for your own corpus and language.

    Parameters
    ----------
    overlap:
        Characters shared between consecutive chunks. Without overlap an answer
        that straddles a boundary appears in neither chunk completely and
        becomes effectively unretrievable. 10-20% of ``chunk_size`` is typical.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(text=text, start=0, end=len(text), index=0, source=source)]

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Snap to a word boundary so chunks do not begin or end mid-word —
        # a chunk starting mid-word embeds poorly.
        if end < len(text):
            boundary = text.rfind(" ", start + step // 2, end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, start=start, end=end, index=index, source=source))
            index += 1

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


def chunk_by_structure(
    text: str,
    max_chunk_size: int = 1024,
    overlap: int = 0,
    source: str | None = None,
) -> list[Chunk]:
    """Split on document structure first, falling back to size.

    Respecting structure — paragraphs, then sentences — beats blind character
    counts, because a chunk that begins mid-sentence carries less coherent
    meaning and embeds worse. Oversized paragraphs still fall back to
    ``chunk_text`` so the size bound always holds.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    offset = 0
    index = 0

    def flush(at: int) -> None:
        nonlocal buffer, buffer_len, index
        if not buffer:
            return
        joined = "\n\n".join(buffer)
        chunks.append(
            Chunk(
                text=joined,
                start=at - len(joined),
                end=at,
                index=index,
                source=source,
            )
        )
        index += 1
        buffer, buffer_len = [], 0

    for para in paragraphs:
        if len(para) > max_chunk_size:
            flush(offset)
            for sub in chunk_text(para, max_chunk_size, overlap, source):
                sub.index = index
                index += 1
                chunks.append(sub)
            offset += len(para) + 2
            continue

        if buffer_len + len(para) > max_chunk_size:
            flush(offset)

        buffer.append(para)
        buffer_len += len(para) + 2
        offset += len(para) + 2

    flush(offset)
    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budgeting. English only, and deliberately crude.

    Real tokenization varies enormously by language — the same sentence can cost
    3-5x more tokens in Thai or Hindi than in English (Alammar ch. 2). For
    anything that matters, count with the actual tokenizer.
    """
    return max(1, len(text) // 4)
