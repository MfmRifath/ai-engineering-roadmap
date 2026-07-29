"""Local state — SQLite, stdlib only.

The one rule: **content never goes in here.** Cards, exercises, notes, and
progress checkboxes all live in the markdown. This database holds only what the
markdown cannot: when you last saw a card and how it went.

That split is what lets you edit a note, `git pull` on another machine, or
regenerate the flashcards, without the app losing your review history or
disagreeing with the repo.

The file is gitignored — it is personal, and it is the only thing here that
cannot be rebuilt from the repository.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from aieng.study.content import REPO_ROOT
from aieng.study.srs import Grade, Schedule, next_due

DEFAULT_DB = REPO_ROOT / ".study" / "study.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS card_state (
    card_id      TEXT PRIMARY KEY,
    repetitions  INTEGER NOT NULL DEFAULT 0,
    ease         REAL    NOT NULL DEFAULT 2.5,
    interval_days INTEGER NOT NULL DEFAULT 0,
    lapses       INTEGER NOT NULL DEFAULT 0,
    due          TEXT,
    last_review  TEXT
);

CREATE TABLE IF NOT EXISTS review_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id   TEXT NOT NULL,
    grade     INTEGER NOT NULL,
    reviewed  TEXT NOT NULL,
    interval_days INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_log_date ON review_log(reviewed);

CREATE TABLE IF NOT EXISTS exercise_state (
    exercise_id TEXT PRIMARY KEY,
    done        INTEGER NOT NULL DEFAULT 0,
    notes       TEXT,
    updated     TEXT
);
"""


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- cards ------------------------------------------------------------

    def schedule_for(self, card_id: str) -> tuple[Schedule, date | None]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM card_state WHERE card_id = ?", (card_id,)).fetchone()
        if row is None:
            return Schedule(), None
        due = date.fromisoformat(row["due"]) if row["due"] else None
        return (
            Schedule(
                repetitions=row["repetitions"],
                ease=row["ease"],
                interval_days=row["interval_days"],
                lapses=row["lapses"],
            ),
            due,
        )

    def all_schedules(self) -> dict[str, tuple[Schedule, date | None]]:
        """One query for the whole deck — the review queue needs every card."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM card_state").fetchall()
        return {
            r["card_id"]: (
                Schedule(
                    repetitions=r["repetitions"],
                    ease=r["ease"],
                    interval_days=r["interval_days"],
                    lapses=r["lapses"],
                ),
                date.fromisoformat(r["due"]) if r["due"] else None,
            )
            for r in rows
        }

    def record_review(
        self, card_id: str, state: Schedule, grade: Grade, today: date | None = None
    ) -> date:
        today = today or date.today()
        due = next_due(state, today)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO card_state
                       (card_id, repetitions, ease, interval_days, lapses, due, last_review)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(card_id) DO UPDATE SET
                       repetitions=excluded.repetitions, ease=excluded.ease,
                       interval_days=excluded.interval_days, lapses=excluded.lapses,
                       due=excluded.due, last_review=excluded.last_review""",
                (
                    card_id,
                    state.repetitions,
                    state.ease,
                    state.interval_days,
                    state.lapses,
                    due.isoformat(),
                    today.isoformat(),
                ),
            )
            conn.execute(
                "INSERT INTO review_log (card_id, grade, reviewed, interval_days)"
                " VALUES (?, ?, ?, ?)",
                (card_id, int(grade), today.isoformat(), state.interval_days),
            )
        return due

    # -- exercises --------------------------------------------------------

    def exercise_done(self) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT exercise_id FROM exercise_state WHERE done = 1").fetchall()
        return {r["exercise_id"] for r in rows}

    def set_exercise_done(self, exercise_id: str, done: bool, notes: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO exercise_state (exercise_id, done, notes, updated)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(exercise_id) DO UPDATE SET
                       done=excluded.done, notes=excluded.notes, updated=excluded.updated""",
                (exercise_id, int(done), notes, datetime.now().isoformat(timespec="seconds")),
            )

    # -- stats ------------------------------------------------------------

    def reviews_on(self, day: date) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM review_log WHERE reviewed = ?", (day.isoformat(),)
            ).fetchone()
        return row["n"]

    def review_history(self, days: int = 30, today: date | None = None) -> list[dict]:
        today = today or date.today()
        start = today - timedelta(days=days - 1)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT reviewed, COUNT(*) AS n FROM review_log"
                " WHERE reviewed >= ? GROUP BY reviewed",
                (start.isoformat(),),
            ).fetchall()
        counts = {r["reviewed"]: r["n"] for r in rows}
        return [
            {
                "date": (d := (start + timedelta(days=i))).isoformat(),
                "count": counts.get(d.isoformat(), 0),
            }
            for i in range(days)
        ]

    def streak(self, today: date | None = None) -> int:
        """Consecutive days with at least one review, ending today or yesterday.

        Yesterday counts so that opening the app in the morning does not show a
        broken streak before the first review of the day.
        """
        today = today or date.today()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT reviewed FROM review_log ORDER BY reviewed DESC"
            ).fetchall()
        seen = {r["reviewed"] for r in rows}
        if not seen:
            return 0

        cursor = today if today.isoformat() in seen else today - timedelta(days=1)
        if cursor.isoformat() not in seen:
            return 0
        streak = 0
        while cursor.isoformat() in seen:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def totals(self) -> dict[str, int]:
        with self._conn() as conn:
            reviews = conn.execute("SELECT COUNT(*) AS n FROM review_log").fetchone()["n"]
            seen = conn.execute("SELECT COUNT(*) AS n FROM card_state").fetchone()["n"]
            lapses = conn.execute(
                "SELECT COALESCE(SUM(lapses), 0) AS n FROM card_state"
            ).fetchone()["n"]
        return {"reviews": reviews, "cards_seen": seen, "lapses": lapses}
