"""Agent memory — Lanham ch. 8.

Long-term agent memory is **RAG with a write path**. The read side is ordinary
retrieval; what is new is deciding what to store, when, how to resolve
contradictions, and when to forget.

Three things this implements that naive memory does not:

* **Supersession** — a newer memory can explicitly retire an older one, so
  retrieval never returns a contradiction for the model to resolve arbitrarily.
* **Recency and importance in the score** — for memory, a recent or important
  item often beats a marginally more similar stale one.
* **Forgetting** — a store that only grows degrades, because noise crowds out
  signal.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

MemoryKind = Literal["semantic", "episodic", "procedural"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    """One remembered thing.

    ``semantic``  — timeless facts ("prefers Python")
    ``episodic``  — timestamped events ("on 3 March, asked about deployment")
    ``procedural``— how to do things; **corrections belong here**, and they are
                    the highest-value memories because they encode a mistake not
                    to repeat.
    """

    content: str
    kind: MemoryKind = "semantic"
    user_id: str = "default"
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    superseded_by: str | None = None
    source_run_id: str | None = None
    embedding: Sequence[float] | None = None

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None

    def age_days(self, now: datetime | None = None) -> float:
        return ((now or _now()) - self.created_at).total_seconds() / 86400


class MemoryStore:
    """An in-memory store with the write-path semantics that matter.

    Deliberately backend-agnostic: swap ``_embed`` and the scan in ``search``
    for a real vector index when the store outgrows RAM. The *policy* — scoring,
    supersession, forgetting — is the part worth getting right, and it does not
    change with the backend.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        *,
        similarity_weight: float = 0.6,
        recency_weight: float = 0.25,
        importance_weight: float = 0.15,
        decay_per_day: float = 0.99,
    ) -> None:
        self._memories: dict[str, Memory] = {}
        self._embed = embed_fn
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.decay_per_day = decay_per_day

    # -- write path -------------------------------------------------------

    def write(self, memory: Memory, supersedes: str | None = None) -> Memory:
        """Store a memory, optionally retiring one it contradicts.

        Explicit supersession is the fix for the classic memory bug: the user
        said Python in March and Go in June, both are stored without timestamps,
        retrieval returns both, and the model picks one at random.
        """
        if self._embed is not None and memory.embedding is None:
            memory.embedding = self._embed(memory.content)

        if supersedes is not None:
            if supersedes not in self._memories:
                raise KeyError(f"cannot supersede unknown memory {supersedes!r}")
            self._memories[supersedes].superseded_by = memory.id

        self._memories[memory.id] = memory
        return memory

    def record_correction(
        self, content: str, *, user_id: str = "default", source_run_id: str | None = None
    ) -> Memory:
        """Corrections are procedural and important — store them as such."""
        return self.write(
            Memory(
                content=content,
                kind="procedural",
                user_id=user_id,
                importance=0.9,
                source_run_id=source_run_id,
            )
        )

    # -- read path --------------------------------------------------------

    def score(
        self,
        memory: Memory,
        query_embedding: Sequence[float] | None,
        now: datetime | None = None,
    ) -> float:
        """Combine similarity, recency, and importance.

        Similarity alone is the right ranking for documents and the wrong one
        for memory: a fact learned yesterday usually beats a marginally more
        similar one from a year ago.
        """
        if not memory.is_active:
            return -1.0  # superseded memories are never retrieved

        similarity = 0.0
        if query_embedding is not None and memory.embedding is not None:
            from aieng.rag.fusion import cosine_similarity

            similarity = cosine_similarity(memory.embedding, query_embedding)

        recency = self.decay_per_day ** memory.age_days(now)
        return (
            self.similarity_weight * similarity
            + self.recency_weight * recency
            + self.importance_weight * memory.importance
        )

    def search(
        self,
        query: str,
        *,
        user_id: str = "default",
        kind: MemoryKind | None = None,
        top_k: int = 5,
    ) -> list[Memory]:
        """Retrieve the most relevant memories.

        ``top_k`` defaults to 5 deliberately. More is distraction — the same
        lost-in-the-middle problem as RAG (Huyen ch. 5, 6).
        """
        query_embedding = self._embed(query) if self._embed is not None else None
        now = _now()

        candidates = [
            m
            for m in self._memories.values()
            if m.is_active and m.user_id == user_id and (kind is None or m.kind == kind)
        ]
        ranked = sorted(
            candidates, key=lambda m: self.score(m, query_embedding, now), reverse=True
        )[:top_k]

        for memory in ranked:
            memory.last_accessed = now
            memory.access_count += 1
        return ranked

    def render(self, memories: Sequence[Memory]) -> str:
        """Format for the prompt, **attributed**.

        Dating each memory lets the model reason about staleness and lets the
        user spot and correct something wrong — neither is possible if memories
        appear as anonymous assertions.
        """
        if not memories:
            return ""
        lines = [f"- {m.content} (learned {m.created_at:%Y-%m-%d})" for m in memories]
        return (
            "Things you have previously learned about this user:\n"
            + "\n".join(lines)
            + "\nIf any of these appear outdated or are contradicted by the "
            "conversation, prefer the newer information and say so."
        )

    # -- forgetting -------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """Hard delete. Users must be able to remove what an agent knows."""
        return self._memories.pop(memory_id, None) is not None

    def prune(
        self,
        *,
        max_age_days: float | None = None,
        min_importance: float = 0.0,
        drop_superseded: bool = True,
    ) -> int:
        """Delete stale, unimportant, or retired memories.

        A store that only grows gets *worse*: retrieval quality falls as the
        index fills with entries nobody needs. Forgetting has to be deliberate.
        """
        now = _now()
        doomed = [
            mid
            for mid, m in self._memories.items()
            if (drop_superseded and not m.is_active)
            or (max_age_days is not None and m.age_days(now) > max_age_days)
            or m.importance < min_importance
        ]
        for mid in doomed:
            del self._memories[mid]
        return len(doomed)

    def consolidate_window(self, days: int = 30) -> list[Memory]:
        """Return recent episodic memories — candidates for reflective summarizing."""
        cutoff = _now() - timedelta(days=days)
        return [
            m
            for m in self._memories.values()
            if m.is_active and m.kind == "episodic" and m.created_at >= cutoff
        ]

    # -- inspection -------------------------------------------------------

    def all(self, *, user_id: str | None = None, include_inactive: bool = False):
        return [
            m
            for m in self._memories.values()
            if (include_inactive or m.is_active) and (user_id is None or m.user_id == user_id)
        ]

    def __len__(self) -> int:
        return sum(1 for m in self._memories.values() if m.is_active)

    def __contains__(self, memory_id: object) -> bool:
        return memory_id in self._memories
