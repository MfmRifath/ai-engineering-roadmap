"""The study app — a local FastAPI server with no external dependencies at runtime.

No LLM, no API keys, no network calls. Everything it serves is derived from the
markdown in this repository plus your own review history in ``.study/study.db``.

Run it:

    make study          # or: python -m aieng.study

Then open http://127.0.0.1:8765.

Binding to 127.0.0.1 is deliberate — this reads and writes files in your
repository, including ROADMAP.md, and has no authentication because it is not
meant to be reachable from anywhere else.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from aieng.study import content, srs
from aieng.study.store import Store

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        'The study app needs FastAPI. Install it with:\n\n    pip install -e ".[study]"\n'
    ) from exc

WEB = Path(__file__).parent / "web"

app = FastAPI(title="AI Engineering Roadmap — Study", docs_url="/api/docs")
store = Store()


class ReviewIn(BaseModel):
    grade: int


class ToggleIn(BaseModel):
    code: str
    done: bool


class ExerciseIn(BaseModel):
    done: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    phases = content.load_roadmap()
    notes = content.all_notes()
    cards = content.all_cards()
    schedules = store.all_schedules()
    today = date.today()

    due = 0
    new = 0
    for card in cards:
        _, due_date = schedules.get(card.id, (srs.Schedule(), None))
        if due_date is None:
            new += 1
        elif srs.is_due(due_date, today):
            due += 1

    done_codes = {t.code for p in phases for t in p.tasks if t.done}
    remaining_hours = 0.0
    code_by_key = _code_by_key(phases)
    for note in notes:
        if code_by_key.get(note.key) not in done_codes:
            remaining_hours += note.est_hours

    return {
        "phases": [
            {
                "name": p.name,
                "done": p.done,
                "total": p.total,
                "pct": round(p.pct, 1),
                "tasks": [
                    {
                        "code": t.code,
                        "title": t.title,
                        "done": t.done,
                        "note_path": t.note_path,
                        "key": _key_from_path(t.note_path),
                    }
                    for t in p.tasks
                ],
            }
            for p in phases
        ],
        "chapters_done": sum(p.done for p in phases),
        "chapters_total": sum(p.total for p in phases),
        "cards_total": len(cards),
        "cards_due": due,
        "cards_new": new,
        "exercises_total": len(content.all_exercises()),
        "exercises_done": len(store.exercise_done()),
        "remaining_hours": round(remaining_hours, 1),
        "streak": store.streak(today),
        "reviews_today": store.reviews_on(today),
        "history": store.review_history(30, today),
        "totals": store.totals(),
    }


def _code_by_key(phases: list[content.Phase]) -> dict[str, str]:
    out: dict[str, str] = {}
    for phase in phases:
        for task in phase.tasks:
            key = _key_from_path(task.note_path)
            if key:
                out[key] = task.code
    return out


def _key_from_path(path: str) -> str:
    import re

    m = re.search(r"([0-9]{2}-[a-z0-9-]+)/notes/ch(\d{2})\.md", path or "")
    return f"{m.group(1)}/{int(m.group(2))}" if m else ""


# ---------------------------------------------------------------------------
# Review — spaced repetition
# ---------------------------------------------------------------------------


@app.get("/api/review/queue")
def review_queue(limit: int = 20, book: str = "", new_limit: int = 10) -> dict[str, Any]:
    """Cards due today, oldest-due first, then a capped number of new ones.

    Capping new cards matters: introducing 363 cards at once guarantees an
    unmanageable review load in three days' time. This is the single most
    common way people abandon spaced repetition.
    """
    schedules = store.all_schedules()
    today = date.today()

    due_cards: list[tuple[date, content.Card]] = []
    new_cards: list[content.Card] = []

    for card in content.all_cards():
        if book and card.book_slug != book:
            continue
        _, due_date = schedules.get(card.id, (srs.Schedule(), None))
        if due_date is None:
            new_cards.append(card)
        elif srs.is_due(due_date, today):
            due_cards.append((due_date, card))

    due_cards.sort(key=lambda t: t[0])
    queue = [c for _, c in due_cards] + new_cards[:new_limit]

    return {
        "total_due": len(due_cards),
        "total_new": len(new_cards),
        "cards": [_card_payload(c, schedules) for c in queue[:limit]],
    }


def _card_payload(card: content.Card, schedules: dict) -> dict[str, Any]:
    state, due_date = schedules.get(card.id, (srs.Schedule(), None))
    return {
        "id": card.id,
        "question": card.question,
        "answer": card.answer,
        "ref": card.ref,
        "book": card.book_slug,
        "chapter": card.chapter,
        "chapter_title": card.chapter_title,
        "is_new": due_date is None,
        "lapses": state.lapses,
        "ease": round(state.ease, 2),
        "preview": srs.preview(state),
    }


@app.post("/api/review/{card_id}")
def submit_review(card_id: str, body: ReviewIn) -> dict[str, Any]:
    card = next((c for c in content.all_cards() if c.id == card_id), None)
    if card is None:
        raise HTTPException(404, f"no card {card_id}")
    try:
        grade = srs.Grade(body.grade)
    except ValueError as exc:
        raise HTTPException(422, f"grade must be one of {[g.value for g in srs.Grade]}") from exc

    state, _ = store.schedule_for(card_id)
    new_state = srs.review(state, grade)
    due = store.record_review(card_id, new_state, grade)
    return {
        "card_id": card_id,
        "interval_days": new_state.interval_days,
        "interval_label": srs.interval_label(new_state.interval_days),
        "due": due.isoformat(),
        "ease": round(new_state.ease, 2),
    }


# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------


@app.get("/api/exercises")
def exercises(book: str = "", chapter: int = 0, kind: str = "") -> dict[str, Any]:
    done = store.exercise_done()
    items = [
        {
            "id": e.id,
            "text": e.text,
            "kind": e.kind,
            "book": e.book_slug,
            "book_label": e.book_label,
            "chapter": e.chapter,
            "chapter_title": e.chapter_title,
            "done": e.id in done,
        }
        for e in content.all_exercises()
        if (not book or e.book_slug == book)
        and (not chapter or e.chapter == chapter)
        and (not kind or e.kind == kind)
    ]
    return {"total": len(items), "done": sum(1 for i in items if i["done"]), "exercises": items}


@app.post("/api/exercises/{exercise_id}")
def set_exercise(exercise_id: str, body: ExerciseIn) -> dict[str, Any]:
    if not any(e.id == exercise_id for e in content.all_exercises()):
        raise HTTPException(404, f"no exercise {exercise_id}")
    store.set_exercise_done(exercise_id, body.done, body.notes)
    return {"exercise_id": exercise_id, "done": body.done}


@app.post("/api/tests/run")
def run_tests(k: str = "") -> dict[str, Any]:
    """Run the repo's own pytest suite and report the result.

    This is how a "build" exercise gets graded without a model: your code
    either passes the tests in ``tests/`` or it does not. Deterministic,
    honest, and already written.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-x"]
    if k:
        cmd += ["-k", k]
    proc = subprocess.run(cmd, cwd=content.REPO_ROOT, capture_output=True, text=True, timeout=600)
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-25:]
    return {"passed": proc.returncode == 0, "output": "\n".join(tail), "command": " ".join(cmd[2:])}


# ---------------------------------------------------------------------------
# Notes, roadmap, graph
# ---------------------------------------------------------------------------


@app.get("/api/notes")
def notes() -> dict[str, Any]:
    return {
        "notes": [
            {
                "key": n.key,
                "book": n.book_slug,
                "book_label": n.book_label,
                "chapter": n.chapter,
                "title": n.title,
                "difficulty": n.difficulty,
                "est_hours": n.est_hours,
                "cards": len(n.cards),
                "exercises": len(n.exercises),
                "code_blocks": n.code_blocks,
                "links": len(n.links_to),
                "path": n.rel_path,
            }
            for n in content.all_notes()
        ]
    }


@app.get("/api/notes/{book}/{chapter}")
def note_detail(book: str, chapter: int) -> dict[str, Any]:
    note = content.note_by_key(f"{book}/{chapter}")
    if note is None:
        raise HTTPException(404, f"no note {book}/{chapter}")
    return {
        "key": note.key,
        "title": note.title,
        "book_label": note.book_label,
        "chapter": note.chapter,
        "difficulty": note.difficulty,
        "est_hours": note.est_hours,
        "tags": note.tags,
        "path": note.rel_path,
        "markdown": note.path.read_text(encoding="utf-8"),
        "links_to": note.links_to,
    }


@app.post("/api/roadmap/toggle")
def toggle(body: ToggleIn) -> dict[str, Any]:
    """Tick a chapter — writes to ROADMAP.md, then regenerates PROGRESS.md."""
    if not content.set_task_done(body.code, body.done):
        raise HTTPException(404, f"no task {body.code}")
    regenerated = False
    script = content.REPO_ROOT / "scripts" / "build_progress.py"
    if script.exists():
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=content.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        regenerated = proc.returncode == 0
    return {"code": body.code, "done": body.done, "progress_regenerated": regenerated}


@app.get("/api/graph")
def graph() -> dict[str, Any]:
    return content.knowledge_graph()


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "notes": len(content.all_notes())})


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> int:
    import uvicorn

    n = len(content.all_notes())
    c = len(content.all_cards())
    e = len(content.all_exercises())
    print("\n  AI Engineering Roadmap — study\n")
    print(f"  {n} notes · {c} flashcards · {e} exercises · no LLM, no network")
    print("\n  http://127.0.0.1:8765\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
