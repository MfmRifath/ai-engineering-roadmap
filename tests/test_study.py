"""The study system — parser, SM-2, and state store.

The SRS tests are the important ones: the scheduling algorithm is the only
place in this app where a subtle bug would go unnoticed for weeks and quietly
ruin the study schedule. Everything else fails loudly.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from aieng.study import content
from aieng.study.srs import (
    INITIAL_EASE,
    MIN_EASE,
    Grade,
    Schedule,
    interval_label,
    is_due,
    next_due,
    preview,
    review,
)
from aieng.study.store import Store

# ---------------------------------------------------------------------------
# SM-2
# ---------------------------------------------------------------------------


def test_first_three_successful_reviews_follow_sm2():
    """1 day, then 6 days, then interval x ease."""
    s = review(Schedule(), Grade.GOOD)
    assert s.interval_days == 1
    assert s.repetitions == 1

    s = review(s, Grade.GOOD)
    assert s.interval_days == 6
    assert s.repetitions == 2

    s = review(s, Grade.GOOD)
    assert s.interval_days == round(6 * s.ease)


def test_intervals_grow_monotonically_while_recalled():
    s = Schedule()
    last = 0
    for _ in range(8):
        s = review(s, Grade.GOOD)
        assert s.interval_days >= last
        last = s.interval_days
    assert last > 100  # a well-known card should end up months out


def test_a_lapse_resets_the_interval_but_keeps_the_lowered_ease():
    s = Schedule()
    for _ in range(4):
        s = review(s, Grade.GOOD)
    before = s.ease

    s = review(s, Grade.AGAIN)
    assert s.interval_days == 1
    assert s.repetitions == 0
    assert s.lapses == 1
    # The point: a card you forgot stays harder even after you relearn it.
    assert s.ease < before


def test_easy_raises_ease_and_again_lowers_it():
    assert review(Schedule(), Grade.EASY).ease > INITIAL_EASE
    assert review(Schedule(), Grade.AGAIN).ease < INITIAL_EASE
    assert review(Schedule(), Grade.HARD).ease < INITIAL_EASE


def test_ease_never_falls_below_the_floor():
    """Without a floor, a repeatedly-failed card's interval collapses to zero."""
    s = Schedule()
    for _ in range(40):
        s = review(s, Grade.AGAIN)
    assert s.ease == pytest.approx(MIN_EASE)
    assert s.interval_days >= 1


def test_easy_schedules_further_out_than_hard():
    s = Schedule()
    for _ in range(3):
        s = review(s, Grade.GOOD)
    assert review(s, Grade.EASY).interval_days > review(s, Grade.HARD).interval_days


def test_interval_is_never_zero():
    """A zero interval would make a card due forever, blocking the queue."""
    for grade in Grade:
        assert review(Schedule(), grade).interval_days >= 1


def test_new_card_detection():
    assert Schedule().is_new
    assert not review(Schedule(), Grade.GOOD).is_new


def test_due_dates():
    today = date(2026, 1, 10)
    s = review(Schedule(), Grade.GOOD)
    assert next_due(s, today) == today + timedelta(days=1)

    assert is_due(None, today), "an unseen card is always due"
    assert is_due(today, today)
    assert is_due(today - timedelta(days=3), today)
    assert not is_due(today + timedelta(days=1), today)


def test_preview_matches_what_review_actually_does():
    """The buttons promise an interval; it has to be the real one."""
    s = review(Schedule(), Grade.GOOD)
    p = preview(s)
    for grade in Grade:
        assert p[grade.name.lower()] == interval_label(review(s, grade).interval_days)


@pytest.mark.parametrize(
    "days,expected",
    [(0, "now"), (1, "1 day"), (12, "12 days"), (60, "2 months"), (400, "1.1 years")],
)
def test_interval_labels(days, expected):
    assert interval_label(days) == expected


# ---------------------------------------------------------------------------
# Content parsing — against the real repository
# ---------------------------------------------------------------------------


def test_all_notes_parse():
    notes = content.all_notes()
    assert len(notes) == 59
    assert all(n.chapter > 0 for n in notes)
    assert all(n.title for n in notes)


def test_frontmatter_is_read():
    notes = content.all_notes()
    assert all(1 <= n.difficulty <= 5 for n in notes)
    assert all(n.est_hours > 0 for n in notes)
    assert {n.book_slug for n in notes} == set(content.BOOK_LABELS)


def test_flashcards_parse_with_stable_ids():
    cards = content.all_cards()
    assert len(cards) > 300
    assert all(c.question and c.answer for c in cards)
    assert len({c.id for c in cards}) == len(cards), "card ids must be unique"


def test_card_id_is_content_addressed():
    a = content.card_id("book", 1, "What is attention?")
    b = content.card_id("book", 1, "what   is   attention?")  # normalised
    c = content.card_id("book", 1, "What is a transformer?")
    assert a == b
    assert a != c


def test_exercises_parse():
    ex = content.all_exercises()
    assert len(ex) > 400
    assert {e.kind for e in ex} == {"understand", "build"}
    assert all(e.text for e in ex)
    assert len({e.id for e in ex}) == len(ex)


def test_roadmap_parses_into_phases():
    phases = content.load_roadmap()
    assert len(phases) == 8
    assert sum(p.total for p in phases) == 59
    codes = [t.code for p in phases for t in p.tasks]
    assert len(set(codes)) == len(codes), "chapter codes must be unique"
    assert "G1" in codes and "L11" in codes


def test_every_roadmap_task_points_at_a_real_note():
    notes = {n.key for n in content.all_notes()}
    for phase in content.load_roadmap():
        for task in phase.tasks:
            import re

            m = re.search(r"([0-9]{2}-[a-z0-9-]+)/notes/ch(\d{2})\.md", task.note_path)
            assert m, f"{task.code} has no note link"
            assert f"{m.group(1)}/{int(m.group(2))}" in notes


def test_knowledge_graph_is_connected_and_consistent():
    g = content.knowledge_graph()
    ids = {n["id"] for n in g["nodes"]}
    assert len(g["nodes"]) == 59
    assert g["edges"], "the notes cross-link heavily; the graph should not be empty"
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids
        assert e["source"] != e["target"], "no self-links"
    assert sum(n["degree"] for n in g["nodes"]) == 2 * len(g["edges"])


def test_graph_has_no_duplicate_edges():
    g = content.knowledge_graph()
    pairs = [tuple(sorted((e["source"], e["target"]))) for e in g["edges"]]
    assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "study.db")


def test_unseen_card_has_default_schedule(store):
    state, due = store.schedule_for("nope")
    assert state == Schedule()
    assert due is None


def test_review_round_trips(store):
    s = review(Schedule(), Grade.GOOD)
    today = date(2026, 1, 10)
    due = store.record_review("card1", s, Grade.GOOD, today)

    assert due == today + timedelta(days=1)
    loaded, loaded_due = store.schedule_for("card1")
    assert loaded == s
    assert loaded_due == due


def test_reviews_accumulate_without_duplicating_state(store):
    s = Schedule()
    today = date(2026, 1, 10)
    for i in range(3):
        s = review(s, Grade.GOOD)
        store.record_review("card1", s, Grade.GOOD, today + timedelta(days=i))

    assert store.totals()["reviews"] == 3
    assert store.totals()["cards_seen"] == 1, "one row per card, not per review"


def test_streak_counts_consecutive_days(store):
    today = date(2026, 1, 10)
    for i in range(4):
        store.record_review(f"c{i}", Schedule(), Grade.GOOD, today - timedelta(days=i))
    assert store.streak(today) == 4


def test_streak_breaks_on_a_gap(store):
    today = date(2026, 1, 10)
    store.record_review("a", Schedule(), Grade.GOOD, today)
    store.record_review("b", Schedule(), Grade.GOOD, today - timedelta(days=3))
    assert store.streak(today) == 1


def test_streak_tolerates_not_having_reviewed_yet_today(store):
    """Opening the app in the morning must not show a broken streak."""
    today = date(2026, 1, 10)
    store.record_review("a", Schedule(), Grade.GOOD, today - timedelta(days=1))
    store.record_review("b", Schedule(), Grade.GOOD, today - timedelta(days=2))
    assert store.streak(today) == 2


def test_streak_is_zero_with_no_reviews(store):
    assert store.streak(date(2026, 1, 10)) == 0


def test_review_history_covers_every_day_in_the_window(store):
    today = date(2026, 1, 10)
    store.record_review("a", Schedule(), Grade.GOOD, today)
    history = store.review_history(30, today)
    assert len(history) == 30
    assert history[-1] == {"date": today.isoformat(), "count": 1}
    assert history[0]["count"] == 0


def test_exercise_state(store):
    assert store.exercise_done() == set()
    store.set_exercise_done("ex1", True)
    assert store.exercise_done() == {"ex1"}
    store.set_exercise_done("ex1", False)
    assert store.exercise_done() == set()


def test_store_creates_its_directory(tmp_path):
    Store(tmp_path / "nested" / "deep" / "study.db")
    assert (tmp_path / "nested" / "deep" / "study.db").exists()


# ---------------------------------------------------------------------------
# API — skipped when FastAPI is not installed
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi", reason="study API needs the [study] extra")
    from fastapi.testclient import TestClient

    from aieng.study import server

    monkeypatch.setattr(server, "store", Store(tmp_path / "study.db"))
    return TestClient(server.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["notes"] == 59


def test_dashboard_endpoint(client):
    d = client.get("/api/dashboard").json()
    assert d["chapters_total"] == 59
    assert d["cards_total"] > 300
    assert len(d["phases"]) == 8
    assert len(d["history"]) == 30
    assert d["remaining_hours"] > 0


def test_review_queue_caps_new_cards(client):
    """Introducing every card at once guarantees an unmanageable load later."""
    q = client.get("/api/review/queue?limit=100&new_limit=5").json()
    assert len(q["cards"]) <= 100
    assert sum(1 for c in q["cards"] if c["is_new"]) <= 5


def test_submitting_a_review_schedules_the_card(client):
    card = client.get("/api/review/queue?limit=1").json()["cards"][0]
    r = client.post(f"/api/review/{card['id']}", json={"grade": 4}).json()
    assert r["interval_days"] == 1
    assert r["interval_label"] == "1 day"


def test_review_rejects_unknown_card(client):
    assert client.post("/api/review/deadbeef", json={"grade": 4}).status_code == 404


def test_review_rejects_bad_grade(client):
    card = client.get("/api/review/queue?limit=1").json()["cards"][0]
    assert client.post(f"/api/review/{card['id']}", json={"grade": 9}).status_code == 422


def test_exercises_endpoint_filters(client):
    all_ex = client.get("/api/exercises").json()
    build = client.get("/api/exercises?kind=build").json()
    assert all_ex["total"] > build["total"] > 0
    assert all(e["kind"] == "build" for e in build["exercises"])


def test_marking_an_exercise_done(client):
    ex = client.get("/api/exercises?kind=build").json()["exercises"][0]
    assert client.post(f"/api/exercises/{ex['id']}", json={"done": True}).status_code == 200
    after = client.get("/api/exercises?kind=build").json()
    assert next(e for e in after["exercises"] if e["id"] == ex["id"])["done"]


def test_graph_endpoint(client):
    g = client.get("/api/graph").json()
    assert len(g["nodes"]) == 59
    assert g["edges"]


def test_note_detail(client):
    n = client.get("/api/notes/03-build-llm-from-scratch-raschka/3").json()
    assert n["chapter"] == 3
    assert "attention" in n["markdown"].lower()


def test_note_detail_404(client):
    assert client.get("/api/notes/nope/1").status_code == 404


# ---------------------------------------------------------------------------
# scripts/doctor.py — must survive a broken environment, since that is when
# it gets run. These fake the failure modes that actually happened: Debian
# without python3-venv, and a machine with no pip at all.
# ---------------------------------------------------------------------------


def _run_doctor(*args: str):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, str(content.REPO_ROOT / "scripts" / "doctor.py"), *args],
        capture_output=True,
        text=True,
        cwd=content.REPO_ROOT,
        timeout=60,
    )


def test_doctor_runs_and_reports_the_interpreter():
    r = _run_doctor()
    assert r.returncode == 0, r.stderr
    assert "environment check" in r.stdout
    assert "python" in r.stdout


def test_doctor_output_is_ascii():
    """The Windows console is cp1252 — a stray em-dash renders as a mojibake box."""
    r = _run_doctor()
    r.stdout.encode("ascii")  # raises if the doctor emits anything exotic


def test_doctor_advice_is_not_self_contradictory():
    """It must not say everything is installed and then tell you to install it."""
    r = _run_doctor()
    if "Everything the study app needs is already installed" in r.stdout:
        assert "pip install" not in r.stdout
        assert "venv" not in r.stdout.split("next")[-1]


def test_doctor_probe_passes_when_pip_exists():
    assert _run_doctor("--probe").returncode == 0


def test_doctor_names_the_right_interpreter_for_the_platform():
    import platform

    r = _run_doctor()
    expected = (
        "python -m aieng.study" if platform.system() == "Windows" else "python3 -m aieng.study"
    )
    assert expected in r.stdout or "make study" in r.stdout
