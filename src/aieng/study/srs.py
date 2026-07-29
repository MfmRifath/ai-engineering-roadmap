"""Spaced repetition — SM-2.

The algorithm behind Anki and SuperMemo, implemented directly because it is
about thirty lines and vendoring a library to avoid thirty lines you should
understand would be silly in a repo about understanding things.

The idea: recall a card just before you would have forgotten it. Each
successful review pushes the next one further out; a lapse pulls it back to the
start. The **ease factor** adapts per card, so cards you find hard come back
more often without you having to say so.

Pure functions of the current state and a grade — no clock, no I/O, no
randomness — which is what makes ``tests/test_study.py`` able to assert on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import IntEnum

MIN_EASE = 1.3
INITIAL_EASE = 2.5


class Grade(IntEnum):
    """What the learner reports after seeing the answer.

    Four buttons rather than SM-2's original 0–5, because finer self-assessment
    is not more accurate — people cannot reliably distinguish six levels of
    "how well did I know that".
    """

    AGAIN = 0  # forgot it
    HARD = 3  # recalled, with difficulty
    GOOD = 4  # recalled correctly
    EASY = 5  # instant


@dataclass(frozen=True)
class Schedule:
    """A card's scheduling state."""

    repetitions: int = 0
    ease: float = INITIAL_EASE
    interval_days: int = 0
    lapses: int = 0

    @property
    def is_new(self) -> bool:
        return self.repetitions == 0 and self.lapses == 0


def review(state: Schedule, grade: Grade) -> Schedule:
    """Apply one review and return the new state.

    >>> s = Schedule()
    >>> s = review(s, Grade.GOOD); s.interval_days
    1
    >>> s = review(s, Grade.GOOD); s.interval_days
    6
    >>> s = review(s, Grade.GOOD); s.interval_days
    15
    >>> review(s, Grade.AGAIN).interval_days      # a lapse resets the interval
    1
    """
    q = int(grade)

    # Ease always adapts, including on a lapse — a card you keep forgetting
    # should keep coming back sooner even after you relearn it.
    ease = state.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    ease = max(MIN_EASE, round(ease, 4))

    if q < 3:
        # Failed. Back to the start, but the lowered ease is retained.
        return Schedule(repetitions=0, ease=ease, interval_days=1, lapses=state.lapses + 1)

    repetitions = state.repetitions + 1
    if repetitions == 1:
        interval = 1
    elif repetitions == 2:
        interval = 6
    else:
        interval = max(1, round(state.interval_days * ease))

    return Schedule(
        repetitions=repetitions,
        ease=ease,
        interval_days=interval,
        lapses=state.lapses,
    )


def next_due(state: Schedule, today: date | None = None) -> date:
    return (today or date.today()) + timedelta(days=state.interval_days)


def is_due(due: date | None, today: date | None = None) -> bool:
    """A card with no due date has never been seen, so it is due."""
    if due is None:
        return True
    return due <= (today or date.today())


def interval_label(days: int) -> str:
    """Human-readable interval, for the button hints."""
    if days <= 0:
        return "now"
    if days == 1:
        return "1 day"
    if days < 30:
        return f"{days} days"
    if days < 365:
        return f"{days / 30:.1f} months".replace(".0", "")
    return f"{days / 365:.1f} years".replace(".0", "")


def preview(state: Schedule) -> dict[str, str]:
    """What each button would do — shown on the buttons themselves.

    Seeing the consequence before choosing makes self-grading far more
    consistent, which matters because the whole algorithm runs on that signal.
    """
    return {g.name.lower(): interval_label(review(state, g).interval_days) for g in Grade}
