"""The interactive study system — local, offline, no LLM.

Turns the repository into something you can *use* rather than only read:

    spaced repetition   over the flashcards already written in the notes
    exercises           the Understand/Build items, tracked, with the real
                        pytest suite as the grader for code tasks
    progress            ticking a chapter writes back to ROADMAP.md
    knowledge graph     the cross-links between chapters, laid out

Design rule: **the markdown is the single source of truth.** This package reads
``books/*/notes/*.md`` and ``ROADMAP.md`` directly and never copies content into
storage. Only your own review history is persisted, in a gitignored SQLite file.
Edit a note and the app reflects it; delete the database and you lose only your
review schedule, nothing else.

No network, no API keys, no model calls.

    make study      # or: python -m aieng.study
"""

from __future__ import annotations

__all__ = ["content", "srs", "store"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
