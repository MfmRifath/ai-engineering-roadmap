"""Read the repository as a study database.

The markdown **is** the database. This module parses it into typed objects on
demand and holds nothing of its own — edit a note, reload, and the app reflects
it. Nothing here duplicates content into storage; only the learner's own review
state is persisted, and that lives in ``store.py``.

Parsed out of the repo:

    ROADMAP.md          phases and per-chapter checkboxes
    books/*/notes/*.md  frontmatter, flashcards, exercises, cross-links
    books/_toc/*.txt    the chapter titles extracted from the real PDFs

No network, no LLM, no external services.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CARD_RE = re.compile(r"<!--\s*card\s*-->(.*?)<!--\s*/card\s*-->", re.DOTALL | re.IGNORECASE)
QA_RE = re.compile(r"^\s*Q:\s*(.*?)\n\s*A:\s*(.*)$", re.DOTALL)
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
PHASE_RE = re.compile(r"^##\s+(Phase\s+\d+[^\n]*)$")
TASK_RE = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$")
# "- [ ] **G1** [The Machine Learning Landscape](books/.../ch01.md)"
TASK_PARTS_RE = re.compile(r"\*\*([A-Z]\d{1,2})\*\*\s*\[([^\]]+)\]\(([^)]+)\)")
XLINK_RE = re.compile(r"\[([^\]]+)\]\((\.\./\.\./[^)]+\.md)\)")
CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

BOOK_LABELS = {
    "01-hands-on-ml-geron": ("Géron", "Hands-On Machine Learning"),
    "02-hands-on-llms-alammar": ("Alammar", "Hands-On Large Language Models"),
    "03-build-llm-from-scratch-raschka": ("Raschka", "Build a LLM (From Scratch)"),
    "04-ai-engineering-huyen": ("Huyen", "AI Engineering"),
    "05-ai-agents-in-action-lanham": ("Lanham", "AI Agents in Action"),
}


def card_id(book_slug: str, chapter: int, question: str) -> str:
    """A stable id for a flashcard.

    Derived from the question text, so editing a question creates a new card
    rather than silently inheriting another card's review history. That is the
    honest behaviour: a reworded question is a different prompt.
    """
    norm = " ".join(question.split()).lower()
    digest = hashlib.sha256(f"{book_slug}|{chapter}|{norm}".encode()).hexdigest()
    return digest[:16]


def exercise_id(book_slug: str, chapter: int, kind: str, index: int) -> str:
    return hashlib.sha256(f"{book_slug}|{chapter}|{kind}|{index}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Card:
    id: str
    question: str
    answer: str
    book_slug: str
    book_label: str
    chapter: int
    chapter_title: str

    @property
    def ref(self) -> str:
        return f"{self.book_label} ch.{self.chapter}"


@dataclass(frozen=True)
class Exercise:
    id: str
    text: str
    kind: str  # "understand" | "build"
    book_slug: str
    book_label: str
    chapter: int
    chapter_title: str


@dataclass
class Note:
    book_slug: str
    book_label: str
    book_title: str
    chapter: int
    title: str
    path: Path
    status: str = "not-started"
    difficulty: int = 3
    est_hours: float = 3.0
    tags: list[str] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    exercises: list[Exercise] = field(default_factory=list)
    links_to: list[str] = field(default_factory=list)  # "book_slug/chapter"
    code_blocks: int = 0

    @property
    def key(self) -> str:
        return f"{self.book_slug}/{self.chapter}"

    @property
    def rel_path(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


@dataclass
class Task:
    """One chapter checkbox in ROADMAP.md."""

    code: str  # "G1", "H10", ...
    title: str
    note_path: str
    done: bool
    phase: str
    line_no: int


@dataclass
class Phase:
    name: str
    tasks: list[Task] = field(default_factory=list)

    @property
    def done(self) -> int:
        return sum(1 for t in self.tasks if t.done)

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def pct(self) -> float:
        return 100.0 * self.done / self.total if self.total else 0.0


def _parse_frontmatter(text: str) -> dict[str, str]:
    """A deliberately tiny YAML subset — the frontmatter is flat scalars only."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.split("#")[0].strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


def _parse_cards(text: str, note: Note) -> list[Card]:
    cards: list[Card] = []
    for block in CARD_RE.findall(text):
        m = QA_RE.match(block.strip())
        if not m:
            continue
        q = " ".join(m.group(1).split())
        a = " ".join(m.group(2).split())
        if not q or not a:
            continue
        cards.append(
            Card(
                id=card_id(note.book_slug, note.chapter, q),
                question=q,
                answer=a,
                book_slug=note.book_slug,
                book_label=note.book_label,
                chapter=note.chapter,
                chapter_title=note.title,
            )
        )
    return cards


def _parse_exercises(text: str, note: Note) -> list[Exercise]:
    """Pull the numbered items under **Understand** and **Build**."""
    out: list[Exercise] = []
    section = re.search(r"^## Exercises\s*$(.*?)^## ", text, re.M | re.S)
    if not section:
        return out
    body = section.group(1)

    for kind, pattern in (
        ("understand", r"\*\*Understand\*\*(.*?)(?=\*\*Build\*\*|\Z)"),
        ("build", r"\*\*Build\*\*(.*?)\Z"),
    ):
        m = re.search(pattern, body, re.S)
        if not m:
            continue
        # Items may wrap onto continuation lines; join them back together.
        items: list[str] = []
        for raw in m.group(1).splitlines():
            if re.match(r"^\d+\.\s", raw.strip()):
                items.append(raw.strip())
            elif items and raw.strip() and raw.startswith(("   ", "\t")):
                items[-1] += " " + raw.strip()
        for i, item in enumerate(items):
            cleaned = re.sub(r"^\d+\.\s*", "", item).strip()
            if cleaned:
                out.append(
                    Exercise(
                        id=exercise_id(note.book_slug, note.chapter, kind, i),
                        text=cleaned,
                        kind=kind,
                        book_slug=note.book_slug,
                        book_label=note.book_label,
                        chapter=note.chapter,
                        chapter_title=note.title,
                    )
                )
    return out


def _parse_links(text: str, book_slug: str) -> list[str]:
    """Cross-book links, normalised to 'book_slug/chapter'."""
    out: set[str] = set()
    for _, target in XLINK_RE.findall(text):
        m = re.search(r"([0-9]{2}-[a-z0-9-]+)/notes/ch(\d{2})\.md", target)
        if m:
            out.add(f"{m.group(1)}/{int(m.group(2))}")
    # same-book links look like "(ch07.md)"
    for target in re.findall(r"\]\((ch\d{2})\.md\)", text):
        out.add(f"{book_slug}/{int(target[2:])}")
    return sorted(out)


def load_note(path: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    book_slug = path.parent.parent.name
    label, book_title = BOOK_LABELS.get(book_slug, (book_slug, book_slug))

    def _num(key: str, default: float) -> float:
        try:
            return float(fm.get(key, default))
        except (TypeError, ValueError):
            return default

    note = Note(
        book_slug=book_slug,
        book_label=label,
        book_title=book_title,
        chapter=int(fm.get("chapter", path.stem.removeprefix("ch") or 0)),
        title=fm.get("title", path.stem),
        path=path,
        status=fm.get("status", "not-started"),
        difficulty=int(_num("difficulty", 3)),
        est_hours=_num("est_hours", 3.0),
        tags=[t.strip() for t in fm.get("tags", "").strip("[]").split(",") if t.strip()],
    )
    note.cards = _parse_cards(text, note)
    note.exercises = _parse_exercises(text, note)
    note.links_to = [k for k in _parse_links(text, book_slug) if k != note.key]
    note.code_blocks = len(CODE_RE.findall(text))
    return note


@lru_cache(maxsize=1)
def _notes_cached(stamp: float) -> tuple[Note, ...]:
    del stamp  # only present to key the cache
    paths = sorted(REPO_ROOT.glob("books/*/notes/ch*.md"))
    return tuple(load_note(p) for p in paths)


def _mtime_stamp() -> float:
    """Newest note mtime, so edits invalidate the cache without a restart."""
    paths = [*REPO_ROOT.glob("books/*/notes/ch*.md"), REPO_ROOT / "ROADMAP.md"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def all_notes() -> list[Note]:
    return list(_notes_cached(_mtime_stamp()))


def all_cards() -> list[Card]:
    return [c for n in all_notes() for c in n.cards]


def all_exercises() -> list[Exercise]:
    return [e for n in all_notes() for e in n.exercises]


def note_by_key(key: str) -> Note | None:
    return next((n for n in all_notes() if n.key == key), None)


# ---------------------------------------------------------------------------
# ROADMAP.md — the progress source of truth
# ---------------------------------------------------------------------------


def load_roadmap() -> list[Phase]:
    path = REPO_ROOT / "ROADMAP.md"
    if not path.exists():
        return []

    phases: list[Phase] = []
    current: Phase | None = None
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        heading = PHASE_RE.match(line)
        if heading:
            current = Phase(name=heading.group(1).strip())
            phases.append(current)
            continue
        task = TASK_RE.match(line)
        if task and current is not None:
            done = task.group(1).lower() == "x"
            body = task.group(2)
            parts = TASK_PARTS_RE.search(body)
            current.tasks.append(
                Task(
                    code=parts.group(1) if parts else body[:6],
                    title=parts.group(2) if parts else body,
                    note_path=parts.group(3) if parts else "",
                    done=done,
                    phase=current.name,
                    line_no=i,
                )
            )
    return [p for p in phases if p.total]


def set_task_done(code: str, done: bool) -> bool:
    """Tick or untick a chapter in ROADMAP.md itself.

    Writing back to the markdown is deliberate: the roadmap stays the single
    source of truth, so progress made in the app shows up in git, in
    PROGRESS.md, and on GitHub. An app with its own private notion of progress
    would immediately disagree with the repo.
    """
    path = REPO_ROOT / "ROADMAP.md"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    for phase in load_roadmap():
        for task in phase.tasks:
            if task.code != code:
                continue
            line = lines[task.line_no]
            updated = (
                re.sub(r"\[[ xX]\]", "[x]" if done else "[ ]", line, count=1)
                if TASK_RE.match(line.rstrip("\n"))
                else line
            )
            if updated == line:
                return False
            lines[task.line_no] = updated
            path.write_text("".join(lines), encoding="utf-8")
            _notes_cached.cache_clear()
            return True
    return False


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------


def knowledge_graph() -> dict:
    """Nodes and edges from the cross-links the notes already carry."""
    notes = all_notes()
    keys = {n.key for n in notes}
    done = {t.code for p in load_roadmap() for t in p.tasks if t.done}

    code_for = {}
    for phase in load_roadmap():
        for task in phase.tasks:
            m = re.search(r"([0-9]{2}-[a-z0-9-]+)/notes/ch(\d{2})\.md", task.note_path)
            if m:
                code_for[f"{m.group(1)}/{int(m.group(2))}"] = task.code

    nodes = [
        {
            "id": n.key,
            "code": code_for.get(n.key, ""),
            "label": f"{n.book_label} {n.chapter}",
            "title": n.title,
            "book": n.book_slug,
            "chapter": n.chapter,
            "difficulty": n.difficulty,
            "cards": len(n.cards),
            "exercises": len(n.exercises),
            "done": code_for.get(n.key, "") in done,
            "degree": 0,
        }
        for n in notes
    ]
    index = {n["id"]: n for n in nodes}

    seen: set[tuple[str, str]] = set()
    edges = []
    for n in notes:
        for target in n.links_to:
            if target not in keys:
                continue
            pair = tuple(sorted((n.key, target)))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({"source": n.key, "target": target})
            index[n.key]["degree"] += 1
            index[target]["degree"] += 1

    return {"nodes": nodes, "edges": edges}
