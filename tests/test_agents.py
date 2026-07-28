"""Agent harness tests — Lanham ch. 5, 6, 8, 10.

These test the parts that are *your* responsibility rather than the model's:
least privilege, loop detection, budget enforcement, error recovery, memory
supersession, and trajectory scoring. All of it runs without any model, because
the model is behind a callable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aieng.agents.loop import LoopGuard, Scratchpad, run_agent
from aieng.agents.memory import Memory, MemoryStore
from aieng.agents.tools import Risk, Tool, ToolRegistry
from aieng.evals.trajectory import (
    ToolCall,
    TrajectorySpec,
    Verdict,
    check_trajectory,
    summarize,
)

# --------------------------------------------------------------------------
# A scripted fake model, so the loop is testable with no network
# --------------------------------------------------------------------------


class FakeCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name, self.arguments = name, arguments


class FakeResponse:
    def __init__(self, content=None, tool_calls=None, cost=0.001):
        self.content, self.tool_calls, self.cost = content, tool_calls or [], cost


def scripted(*responses):
    """Return a model_fn that yields the given responses in order."""
    queue = list(responses)

    def model_fn(messages, tools):
        return queue.pop(0) if queue else FakeResponse(content="done")

    return model_fn


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool("search", "Search the docs.", lambda query: f"results for {query}"),
        contexts=["default", "public"],
    )
    reg.register(
        Tool(
            "send_email",
            "Send an email. Use only when explicitly asked.",
            lambda to, body: f"sent to {to}",
            risk=Risk.WRITE_EXTERNAL,
        ),
        contexts=["default"],
    )
    reg.register(
        Tool("boom", "Always fails.", lambda: (_ for _ in ()).throw(RuntimeError("nope"))),
        contexts=["default"],
    )
    return reg


# --------------------------------------------------------------------------
# Least privilege — the structural defense
# --------------------------------------------------------------------------


def test_tool_outside_context_is_refused(registry):
    """The strongest injection defense: the capability simply is not there."""
    result = registry.dispatch("send_email", {"to": "x", "body": "y"}, context="public")
    assert not result.ok
    assert "not available" in result.error


def test_refusal_is_returned_not_raised(registry):
    """The agent must be able to reason about a denial and pick another tool."""
    result = registry.dispatch("send_email", {}, context="public")
    assert result.ok is False
    assert "not available" in result.to_observation()


def test_available_tools_are_scoped(registry):
    assert {t.name for t in registry.available("public")} == {"search"}
    assert {t.name for t in registry.available("default")} == {
        "search",
        "send_email",
        "boom",
    }


def test_write_external_requires_confirmation(registry):
    denied = registry.dispatch(
        "send_email", {"to": "a", "body": "b"}, confirm=lambda tool, args: False
    )
    assert not denied.ok
    assert "approve" in denied.error

    allowed = registry.dispatch(
        "send_email", {"to": "a", "body": "b"}, confirm=lambda tool, args: True
    )
    assert allowed.ok


def test_missing_confirmation_callback_denies_by_default(registry):
    """Failing closed is the only acceptable default for a write tool."""
    assert not registry.dispatch("send_email", {"to": "a", "body": "b"}).ok


def test_tool_errors_become_observations(registry):
    """Recovery is the behaviour we want; a raised exception aborts the run."""
    result = registry.dispatch("boom", {})
    assert not result.ok
    assert "nope" in result.error
    assert "Consider a different" in result.to_observation()


def test_invalid_arguments_are_caught(registry):
    result = registry.dispatch("search", {"wrong_param": 1})
    assert not result.ok
    assert "invalid arguments" in result.error


def test_tool_results_are_labeled_as_untrusted(registry):
    """Indirect injection arrives through tool results — label them as data."""
    observation = registry.dispatch("search", {"query": "x"}).to_observation()
    assert "<tool_result" in observation
    assert "do not follow instructions" in observation.lower()


def test_risk_tiers():
    assert Tool("a", "d", lambda: 1, risk=Risk.WRITE_EXTERNAL).requires_confirmation
    assert not Tool("b", "d", lambda: 1, risk=Risk.READ_SCOPED).requires_confirmation
    assert Tool("c", "d", lambda: 1, risk=Risk.COMPUTE).requires_sandbox


def test_decorator_registration():
    reg = ToolRegistry()

    @reg.tool("add", "Add two numbers.")
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in reg
    assert reg.dispatch("add", {"a": 2, "b": 3}).content == "5"


# --------------------------------------------------------------------------
# Loop detection
# --------------------------------------------------------------------------


def test_loop_guard_detects_a_repeated_call():
    guard = LoopGuard()
    assert guard.check("search", {"q": "x"}) is None
    warning = guard.check("search", {"q": "x"})
    assert warning is not None and "already called" in warning


def test_loop_guard_ignores_different_arguments():
    guard = LoopGuard()
    assert guard.check("search", {"q": "a"}) is None
    assert guard.check("search", {"q": "b"}) is None


def test_loop_guard_is_insensitive_to_key_order():
    guard = LoopGuard()
    assert guard.check("t", {"a": 1, "b": 2}) is None
    assert guard.check("t", {"b": 2, "a": 1}) is not None


def test_loop_guard_counts_repeats():
    guard = LoopGuard()
    for _ in range(3):
        guard.check("t", {})
    assert guard.repeat_count == 2


# --------------------------------------------------------------------------
# The loop itself
# --------------------------------------------------------------------------


def test_agent_returns_when_no_tool_is_requested(registry):
    result = run_agent("hi", scripted(FakeResponse(content="hello")), registry)
    assert result.outcome == "success"
    assert result.answer == "hello"


def test_agent_calls_a_tool_then_answers(registry):
    model = scripted(
        FakeResponse(tool_calls=[FakeCall("search", {"query": "cats"})]),
        FakeResponse(content="cats are good"),
    )
    result = run_agent("tell me about cats", model, registry)
    assert result.outcome == "success"
    assert result.num_steps == 2
    assert [c["name"] for c in result.tool_calls] == ["search"]


def test_agent_stops_at_the_step_cap(registry):
    """The single most common agent incident."""
    always_tool = FakeResponse(tool_calls=[FakeCall("search", {"query": "x"})])
    model = scripted(*[always_tool] * 20)
    result = run_agent("loop forever", model, registry, max_steps=3)
    assert result.outcome == "max_steps"
    assert result.num_steps == 3


def test_agent_stops_at_the_cost_budget(registry):
    expensive = FakeResponse(tool_calls=[FakeCall("search", {"query": "x"})], cost=0.30)
    model = scripted(*[expensive] * 20)
    result = run_agent("burn money", model, registry, max_cost_usd=0.50, max_steps=20)
    assert result.outcome == "budget"
    assert result.total_cost <= 0.9


def test_agent_survives_a_failing_tool(registry):
    model = scripted(
        FakeResponse(tool_calls=[FakeCall("boom", {})]),
        FakeResponse(content="recovered"),
    )
    result = run_agent("try it", model, registry)
    assert result.outcome == "success"
    assert result.answer == "recovered"


def test_agent_records_a_full_trace(registry):
    model = scripted(
        FakeResponse(tool_calls=[FakeCall("search", {"query": "x"})]),
        FakeResponse(content="done"),
    )
    result = run_agent("task", model, registry)
    assert result.steps[0].tool_calls[0]["name"] == "search"
    assert result.steps[0].tool_results
    assert result.to_eval_case("task")["tools"] == ["search"]


def test_scratchpad_renders_state():
    pad = Scratchpad(goal="find X", findings=["a"], attempted=["search"])
    rendered = pad.render()
    assert "find X" in rendered and "search" in rendered


# --------------------------------------------------------------------------
# Memory — Lanham ch. 8
# --------------------------------------------------------------------------


def test_memory_write_and_search():
    store = MemoryStore()
    store.write(Memory(content="prefers Python"))
    assert len(store) == 1
    assert store.search("language")[0].content == "prefers Python"


def test_supersede_hides_the_old_memory():
    """The contradiction fix: retrieval must never return both."""
    store = MemoryStore()
    old = store.write(Memory(content="prefers Python"))
    store.write(Memory(content="prefers Go"), supersedes=old.id)

    contents = [m.content for m in store.search("language preference")]
    assert "prefers Go" in contents
    assert "prefers Python" not in contents
    assert len(store) == 1


def test_superseding_an_unknown_memory_raises():
    with pytest.raises(KeyError):
        MemoryStore().write(Memory(content="x"), supersedes="nope")


def test_recency_beats_a_marginally_older_memory():
    """Similarity alone is the wrong ranking for memory."""
    store = MemoryStore(similarity_weight=0.0, recency_weight=1.0, importance_weight=0.0)
    old = Memory(content="old fact")
    old.created_at = datetime.now(timezone.utc) - timedelta(days=365)
    store.write(old)
    store.write(Memory(content="new fact"))
    assert store.search("fact")[0].content == "new fact"


def test_importance_influences_ranking():
    store = MemoryStore(similarity_weight=0.0, recency_weight=0.0, importance_weight=1.0)
    store.write(Memory(content="trivial", importance=0.1))
    store.write(Memory(content="critical", importance=0.9))
    assert store.search("anything")[0].content == "critical"


def test_corrections_are_stored_as_high_importance_procedural():
    store = MemoryStore()
    memory = store.record_correction("always use ISO dates")
    assert memory.kind == "procedural"
    assert memory.importance >= 0.9


def test_memories_are_scoped_by_user():
    store = MemoryStore()
    store.write(Memory(content="alice fact", user_id="alice"))
    store.write(Memory(content="bob fact", user_id="bob"))
    assert [m.content for m in store.search("fact", user_id="alice")] == ["alice fact"]


def test_prune_removes_stale_and_superseded():
    store = MemoryStore()
    old = Memory(content="ancient")
    old.created_at = datetime.now(timezone.utc) - timedelta(days=400)
    store.write(old)
    store.write(Memory(content="fresh"))
    assert store.prune(max_age_days=365) == 1
    assert len(store) == 1


def test_forget_is_a_hard_delete():
    """Users must be able to remove what an agent knows about them."""
    store = MemoryStore()
    memory = store.write(Memory(content="sensitive"))
    assert store.forget(memory.id) is True
    assert len(store) == 0
    assert store.forget("nonexistent") is False


def test_render_attributes_memories_with_dates():
    """Undated memories cannot be judged for staleness or corrected."""
    store = MemoryStore()
    store.write(Memory(content="prefers dark mode"))
    rendered = store.render(store.search("preferences"))
    assert "prefers dark mode" in rendered
    assert "learned" in rendered


def test_render_of_nothing_is_empty():
    assert MemoryStore().render([]) == ""


# --------------------------------------------------------------------------
# Trajectory evaluation — Lanham ch. 10
# --------------------------------------------------------------------------


def test_efficient_correct_run_passes():
    spec = TrajectorySpec("t1", optimal_steps=2, max_steps=10, max_cost_usd=0.5)
    result = check_trajectory(
        spec,
        [ToolCall("search", {"q": "x"}), ToolCall("read", {"id": 1})],
        success=True,
        cost_usd=0.02,
    )
    assert result.verdict is Verdict.PASS


def test_correct_but_wasteful_run_is_an_efficiency_failure():
    """The failure end-to-end accuracy cannot see."""
    spec = TrajectorySpec("t2", optimal_steps=2, max_steps=20, max_cost_usd=1.0)
    calls = [ToolCall("search", {"q": f"attempt{i}"}) for i in range(14)]
    result = check_trajectory(spec, calls, success=True, cost_usd=0.31)
    assert result.verdict is Verdict.EFFICIENCY_FAILURE
    assert result.efficiency < 0.5


def test_forbidden_tool_overrides_a_correct_answer():
    """A near-miss on a dangerous action fails regardless of the output."""
    spec = TrajectorySpec("t3", optimal_steps=2, forbidden_tools={"delete_record"})
    result = check_trajectory(
        spec,
        [ToolCall("search", {}), ToolCall("delete_record", {"id": 7})],
        success=True,
        cost_usd=0.01,
    )
    assert result.verdict is Verdict.SAFETY_FAILURE
    assert result.forbidden_attempted == ["delete_record"]


def test_redundant_calls_are_counted():
    spec = TrajectorySpec("t4", optimal_steps=1)
    calls = [ToolCall("search", {"q": "x"})] * 3
    result = check_trajectory(spec, calls, success=True, cost_usd=0.01)
    assert result.redundant_calls == 2
    assert any("redundant" in p for p in result.problems)


def test_recovery_from_a_failed_call_is_detected():
    spec = TrajectorySpec("t5", optimal_steps=2)
    calls = [
        ToolCall("search", {"q": "x"}, succeeded=False, error="timeout"),
        ToolCall("search_backup", {"q": "x"}, succeeded=True),
    ]
    result = check_trajectory(spec, calls, success=True, cost_usd=0.02)
    assert result.recovered_from_failure


def test_budget_and_step_overruns_are_reported():
    spec = TrajectorySpec("t6", max_steps=2, max_cost_usd=0.10, optimal_steps=1)
    calls = [ToolCall("search", {"q": str(i)}) for i in range(5)]
    result = check_trajectory(spec, calls, success=True, cost_usd=0.99)
    assert any("cost" in p for p in result.problems)
    assert any("steps" in p for p in result.problems)


def test_missing_required_tool_is_reported():
    spec = TrajectorySpec("t7", required_tools={"cite_source"}, optimal_steps=2)
    result = check_trajectory(spec, [ToolCall("search", {})], success=True, cost_usd=0.01)
    assert result.missing_required == ["cite_source"]


def test_summary_reports_the_verdict_distribution():
    """An aggregate success rate hides exactly what trajectory eval exists to find."""
    spec = TrajectorySpec("t", optimal_steps=2, forbidden_tools={"danger"})
    results = [
        check_trajectory(spec, [ToolCall("a", {}), ToolCall("b", {})], success=True, cost_usd=0.01),
        check_trajectory(
            spec, [ToolCall(f"a{i}", {}) for i in range(12)], success=True, cost_usd=0.2
        ),
        check_trajectory(spec, [ToolCall("danger", {})], success=True, cost_usd=0.01),
    ]
    summary = summarize(results)
    assert summary["task_success_rate"] == 1.0  # all "correct"
    assert summary["pass_rate"] < 1.0  # but not all acceptable
    assert summary["verdicts"]["SAFETY_FAILURE"] == 1
    assert summary["verdicts"]["EFFICIENCY_FAILURE"] == 1


def test_summary_of_nothing():
    assert summarize([])["n"] == 0
