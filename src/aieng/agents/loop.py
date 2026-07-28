"""The agent loop and its guards — Lanham ch. 1, 6; Huyen ch. 6.

The loop itself is twenty lines. Everything here is making those twenty lines
survivable in production: step caps, cost budgets, timeouts, loop detection,
and graceful degradation.

The classic agent incident is not a wrong answer — it is a loop and a runaway
bill. Every bound in this module exists because of that.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from aieng.agents.tools import ToolRegistry, ToolResult


class LoopGuard:
    """Detect repeated identical tool calls.

    Two identical calls mean the agent did not learn from the first result. The
    right response is to **tell it** — it can often recover — rather than to
    abort, which wastes the run and hides the failure.
    """

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold
        self.seen: Counter[tuple[str, str]] = Counter()

    def check(self, name: str, arguments: dict) -> str | None:
        key = (name, json.dumps(arguments, sort_keys=True, default=str))
        self.seen[key] += 1
        if self.seen[key] >= self.threshold:
            return (
                f"You have already called {name} with these exact arguments and "
                f"received a result. Repeating it will not produce anything new — "
                f"try a different tool, different arguments, or answer with what "
                f"you have."
            )
        return None

    @property
    def repeat_count(self) -> int:
        return sum(c - 1 for c in self.seen.values() if c > 1)


@dataclass
class Scratchpad:
    """Compact structured state, rendered fresh each turn.

    Relying on raw message history to carry state is expensive (it grows every
    step, and you resend all of it) and unreliable. A scratchpad is cheaper,
    more durable, and — usefully — inspectable when debugging.
    """

    goal: str
    findings: list[str] = field(default_factory=list)
    attempted: list[str] = field(default_factory=list)
    blocked_by: str | None = None

    def render(self) -> str:
        return (
            f"GOAL: {self.goal}\n"
            f"FOUND SO FAR: {'; '.join(self.findings) or 'nothing yet'}\n"
            f"ALREADY TRIED: {'; '.join(self.attempted) or 'nothing yet'}\n"
            f"BLOCKED BY: {self.blocked_by or 'nothing'}"
        )


Outcome = Literal["success", "max_steps", "budget", "timeout", "error"]


@dataclass
class StepTrace:
    """One step of a run. The unit of trajectory evaluation."""

    step: int
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    """A complete run. Every field here exists because you need it to debug.

    A chatbot failure is one call you can reproduce. An agent failure is a
    *trajectory* — the wrong tool at step 3 caused the wrong result at step 4
    which caused the loop at steps 5 through 9. Without the full trace you are
    guessing, and with it every failed run becomes an eval case.
    """

    outcome: Outcome
    answer: str | None
    steps: list[StepTrace] = field(default_factory=list)
    scratchpad: Scratchpad | None = None
    redundant_calls: int = 0

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> list[dict]:
        return [c for s in self.steps for c in s.tool_calls]

    def to_eval_case(self, task: str) -> dict:
        """Turn a run into a regression test — how the eval set grows."""
        return {
            "task": task,
            "outcome": self.outcome,
            "steps": self.num_steps,
            "cost_usd": round(self.total_cost, 6),
            "tools": [c["name"] for c in self.tool_calls],
        }


def run_agent(
    task: str,
    model_fn: Any,
    registry: ToolRegistry,
    *,
    context: str = "default",
    max_steps: int = 10,
    max_cost_usd: float = 0.50,
    timeout_s: float = 120.0,
    confirm: Any = None,
) -> AgentResult:
    """Run an agent loop with every guard in place.

    ``model_fn(messages, tools) -> response`` where ``response`` exposes
    ``.content``, ``.tool_calls`` (each with ``.name`` and ``.arguments``), and
    optionally ``.cost``. Keeping the model behind a callable makes the loop
    testable with a scripted fake — see ``tests/test_agent_loop.py``.

    Degrades rather than raising: a partial answer with an explanation is more
    useful to a caller than a stack trace.
    """
    guard = LoopGuard()
    pad = Scratchpad(goal=task)
    messages: list[dict] = [{"role": "user", "content": task}]
    steps: list[StepTrace] = []
    started = time.monotonic()
    cost = 0.0

    for step_no in range(max_steps):
        if time.monotonic() - started > timeout_s:
            return AgentResult("timeout", None, steps, pad, guard.repeat_count)
        if cost > max_cost_usd:
            return AgentResult("budget", None, steps, pad, guard.repeat_count)

        step_start = time.monotonic()
        response = model_fn(messages, registry.schemas(context))
        cost += getattr(response, "cost", 0.0)

        trace = StepTrace(
            step=step_no,
            cost_usd=getattr(response, "cost", 0.0),
            latency_ms=(time.monotonic() - step_start) * 1000,
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            steps.append(trace)
            return AgentResult("success", response.content, steps, pad, guard.repeat_count)

        messages.append({"role": "assistant", "content": response.content or ""})

        for call in tool_calls:
            trace.tool_calls.append({"name": call.name, "arguments": call.arguments})

            if (warning := guard.check(call.name, call.arguments)) is not None:
                trace.tool_results.append(warning)
                messages.append({"role": "tool", "content": warning})
                continue

            pad.attempted.append(call.name)
            result: ToolResult = registry.dispatch(
                call.name, call.arguments, context=context, confirm=confirm
            )
            observation = result.to_observation()
            trace.tool_results.append(observation)
            messages.append({"role": "tool", "content": observation})

            if result.ok:
                pad.findings.append(f"{call.name}: {result.content[:120]}")
            else:
                pad.blocked_by = result.error

        steps.append(trace)

    return AgentResult("max_steps", None, steps, pad, guard.repeat_count)
