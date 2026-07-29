"""Reading and coding — the two surfaces where you actually study.

The most important test here is ``test_starter_code_fails_every_challenge``:
a challenge whose checks pass before you have written anything teaches nothing,
and that failure mode is silent.
"""

from __future__ import annotations

import pytest

from aieng.study import challenges, render
from aieng.study.runner import build_script, run_challenge, run_scratch

# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------


def test_challenge_set_is_sane():
    assert len(challenges.CHALLENGES) >= 15
    ids = [c.id for c in challenges.CHALLENGES]
    assert len(set(ids)) == len(ids), "challenge ids must be unique"
    assert {c.phase for c in challenges.CHALLENGES} <= {1, 2, 3, 4, 5, 6, 7, 8}


def test_every_challenge_is_well_formed():
    for c in challenges.CHALLENGES:
        assert c.prompt.strip(), f"{c.id} has no prompt"
        assert c.starter.strip(), f"{c.id} has no starter code"
        assert c.tests, f"{c.id} has no tests"
        assert c.solution.strip(), f"{c.id} has no reference solution"
        assert 1 <= c.difficulty <= 5


def test_every_challenge_has_a_hidden_test():
    """Visible tests alone can be gamed by returning the expected value."""
    for c in challenges.CHALLENGES:
        assert any(t.hidden for t in c.tests), f"{c.id} has no hidden test"


@pytest.mark.parametrize("challenge", challenges.CHALLENGES, ids=lambda c: c.id)
def test_reference_solution_passes(challenge):
    result = run_challenge(challenge.solution, challenge)
    assert result.ok, (
        f"{challenge.id}: {result.passed}/{result.total} — {result.error}\n"
        + "\n".join(f"  {r.name}: {r.error}" for r in result.results if not r.passed)
    )


@pytest.mark.parametrize("challenge", challenges.CHALLENGES, ids=lambda c: c.id)
def test_starter_code_fails(challenge):
    """The starter must not pass. A challenge you have already solved is not one."""
    assert not run_challenge(challenge.starter, challenge).ok


def test_hidden_test_catches_the_naive_overflowing_softmax():
    """The exact mistake the challenge exists to teach must actually be caught."""
    naive = (
        "import math\n"
        "def softmax(logits):\n"
        "    e = [math.exp(x) for x in logits]\n"
        "    s = sum(e)\n"
        "    return [v / s for v in e]\n"
    )
    result = run_challenge(naive, challenges.get("softmax"))
    assert not result.ok
    failed = {r.name for r in result.results if not r.passed}
    assert "does not overflow on large logits" in failed


def test_hidden_test_catches_unbiased_variance_in_layer_norm():
    wrong = (
        "def layer_norm(x, eps=1e-5):\n"
        "    m = sum(x) / len(x)\n"
        "    v = sum((q - m) ** 2 for q in x) / (len(x) - 1)\n"
        "    return [(q - m) / (v + eps) ** 0.5 for q in x]\n"
    )
    result = run_challenge(wrong, challenges.get("layer_norm"))
    assert not result.ok
    assert "uses biased variance" in {r.name for r in result.results if not r.passed}


def test_get_unknown_challenge():
    assert challenges.get("nope") is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_syntax_error_is_reported_not_crashed():
    result = run_challenge("def broken(:\n  pass", challenges.get("softmax"))
    assert not result.ok
    assert "SyntaxError" in result.error


def test_infinite_loop_is_killed():
    result = run_scratch("while True:\n    pass")
    assert not result.ok
    assert "imed out" in result.error


def test_scratch_captures_stdout():
    result = run_scratch("print('hello')")
    assert result.ok
    assert "hello" in result.stdout


def test_scratch_reports_a_runtime_error():
    result = run_scratch("raise ValueError('boom')")
    assert not result.ok
    assert "boom" in result.error


def test_output_is_capped():
    """A print loop must not be able to exhaust the server's memory."""
    result = run_scratch("for _ in range(200000):\n    print('x' * 80)")
    assert len(result.stdout) <= 10_000


def test_one_failing_check_does_not_hide_the_others():
    """Each test is wrapped separately, so you see all results, not just the first."""
    result = run_challenge(
        "def softmax(logits):\n    return [0.0] * len(logits)\n", challenges.get("softmax")
    )
    assert result.total == len(challenges.get("softmax").tests)


def test_build_script_includes_user_code_and_every_test():
    c = challenges.get("softmax")
    script = build_script("def softmax(x):\n    return x\n", c)
    assert "def softmax(x):" in script
    for i in range(len(c.tests)):
        assert f"def _t{i}():" in script


def test_runner_result_serialises():
    d = run_challenge(
        challenges.get("compounding").solution, challenges.get("compounding")
    ).to_dict()
    assert d["ok"] is True
    assert set(d) >= {"ok", "passed", "total", "results", "stdout", "error", "duration_ms"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# markdown-it-py ships in the [study] extra. On a minimal `pip install -e ".[dev]"`
# these skip rather than fail — a contributor running the fast suite should not
# see red for a feature they did not install. CI installs [study], so they run there.
needs_markdown = pytest.mark.skipif(
    not render.markdown_available(),
    reason='needs markdown-it-py: pip install -e ".[study]"',
)


@needs_markdown
def test_markdown_renders_to_html():
    html = render.render_note("# Title\n\nSome **bold** text.\n", "01-hands-on-ml-geron")
    assert "<h1>" in html or "Title" in html
    assert "bold" in html


def test_frontmatter_is_stripped():
    md = "---\nbook: X\nchapter: 1\n---\n\n# Real heading\n"
    html = render.render_note(md, "01-hands-on-ml-geron")
    assert "book: X" not in html


@needs_markdown
def test_cross_book_links_become_in_app_routes():
    md = "See [A3](../../02-hands-on-llms-alammar/notes/ch03.md) for this.\n"
    html = render.render_note(md, "01-hands-on-ml-geron")
    assert "#/read/02-hands-on-llms-alammar/3" in html
    assert ".md" not in html


@needs_markdown
def test_same_book_links_become_in_app_routes():
    html = render.render_note("Back to [ch7](ch07.md).\n", "04-ai-engineering-huyen")
    assert "#/read/04-ai-engineering-huyen/7" in html


@needs_markdown
def test_diagram_paths_are_rewritten_to_the_app_mount():
    md = "![a diagram](../../../assets/kv-cache.svg)\n"
    html = render.render_note(md, "02-hands-on-llms-alammar")
    assert 'src="/assets/kv-cache.svg"' in html


@needs_markdown
def test_tables_render_as_tables():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert "<table>" in render.render_note(md, "x")


def test_toc_extracts_headings_but_not_code_comments():
    md = "## Real heading\n\n```python\n# not a heading\n```\n\n### Sub\n"
    toc = render.note_toc(md)
    assert [t["title"] for t in toc] == ["Real heading", "Sub"]
    assert [t["level"] for t in toc] == [2, 3]


def test_every_real_note_renders():
    """The renderer must survive all 59 notes, not just a sample."""
    from aieng.study import content

    for note in content.all_notes():
        html = render.render_note(note.path.read_text(encoding="utf-8"), note.book_slug)
        assert len(html) > 500, f"{note.key} rendered suspiciously short"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aieng.study import server
    from aieng.study.store import Store

    monkeypatch.setattr(server, "store", Store(tmp_path / "study.db"))
    return TestClient(server.app)


def test_library_lists_every_book(client):
    d = client.get("/api/library").json()
    assert len(d["books"]) == 5
    assert sum(len(b["chapters"]) for b in d["books"]) == 59


def test_read_endpoint(client):
    n = client.get("/api/read/03-build-llm-from-scratch-raschka/3").json()
    assert n["chapter"] == 3
    assert n["html"]
    assert n["toc"]
    assert n["next"]["chapter"] == 4


def test_read_first_chapter_has_no_prev(client):
    assert client.get("/api/read/01-hands-on-ml-geron/1").json()["prev"] is None


def test_read_404(client):
    assert client.get("/api/read/nope/1").status_code == 404


def test_challenge_endpoints(client):
    listing = client.get("/api/challenges").json()["challenges"]
    assert len(listing) >= 15

    c = client.get("/api/challenges/softmax").json()
    assert c["starter"]
    assert c["hidden_count"] >= 1
    # Hidden test bodies must not be served to the browser.
    assert all("overflow" not in t["name"] for t in c["tests"])


def test_running_a_correct_solution_marks_it_done(client):
    good = (
        "import math\n"
        "def softmax(logits):\n"
        "    m = max(logits)\n"
        "    e = [math.exp(x - m) for x in logits]\n"
        "    s = sum(e)\n"
        "    return [v / s for v in e]\n"
    )
    r = client.post("/api/challenges/softmax/run", json={"code": good}).json()
    assert r["ok"] and r["passed"] == r["total"]

    listing = client.get("/api/challenges").json()["challenges"]
    assert next(c for c in listing if c["id"] == "softmax")["done"]


def test_running_a_wrong_solution_does_not_mark_it_done(client):
    r = client.post(
        "/api/challenges/softmax/run", json={"code": "def softmax(x):\n    return x\n"}
    ).json()
    assert not r["ok"]
    listing = client.get("/api/challenges").json()["challenges"]
    assert not next(c for c in listing if c["id"] == "softmax")["done"]


def test_challenge_run_404(client):
    assert client.post("/api/challenges/nope/run", json={"code": "x=1"}).status_code == 404


def test_solution_endpoint(client):
    assert "def softmax" in client.get("/api/challenges/softmax/solution").json()["solution"]


def test_scratch_endpoint(client):
    r = client.post("/api/scratch/run", json={"code": "print(6*7)"}).json()
    assert r["ok"] and "42" in r["stdout"]


# ---------------------------------------------------------------------------
# The one endpoint that writes to the repository
# ---------------------------------------------------------------------------


def test_toggling_a_chapter_round_trips_through_roadmap_md():
    """`set_task_done` edits the real ROADMAP.md, which is the whole point —
    the app and the repo must never hold different ideas of your progress.

    That also makes it the one function a careless test could use to mutate the
    working tree, so this restores the original bytes in a finally block and
    asserts they came back.
    """
    from aieng.study import content

    roadmap = content.REPO_ROOT / "ROADMAP.md"
    original = roadmap.read_bytes()

    try:
        phases = content.load_roadmap()
        code = phases[0].tasks[0].code
        assert not phases[0].tasks[0].done, "expected a clean roadmap"

        assert content.set_task_done(code, True) is True
        reloaded = {t.code: t.done for p in content.load_roadmap() for t in p.tasks}
        assert reloaded[code] is True
        # Exactly one box moved.
        assert sum(reloaded.values()) == 1

        assert content.set_task_done(code, False) is True
        assert not any(t.done for p in content.load_roadmap() for t in p.tasks)
    finally:
        roadmap.write_bytes(original)
        content._notes_cached.cache_clear()

    assert roadmap.read_bytes() == original, "ROADMAP.md was not restored"


def test_toggling_an_unknown_chapter_changes_nothing():
    from aieng.study import content

    roadmap = content.REPO_ROOT / "ROADMAP.md"
    before = roadmap.read_bytes()
    assert content.set_task_done("ZZ99", True) is False
    assert roadmap.read_bytes() == before
