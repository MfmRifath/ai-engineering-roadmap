# p5 — Agent platform

**After Phase 8 · Draws on [L5](../../books/05-ai-agents-in-action-lanham/notes/ch05.md)–[L11](../../books/05-ai-agents-in-action-lanham/notes/ch11.md)**

A tool-using agent with memory that **recovers from failures**, plus the tracing
and evaluation that make it operable.

## Why this project

Anyone can get an agent to work on the happy path in an afternoon. The gap between
that and something you would put in front of users is entirely: what happens when
a tool fails, when the model loops, when someone injects instructions into a
retrieved document, and how you find out which step went wrong.

That gap is this project.

## Definition of done

**The loop**
- [ ] Built on [`aieng.agents.loop`](../../src/aieng/agents/loop.py) with step cap,
      cost budget, wall-clock timeout, and loop detection — all enforced
- [ ] Degrades gracefully: a partial answer with an explanation, never a traceback
- [ ] Tool errors return as observations so the agent can adapt

**Tools**
- [ ] A registry with **per-context least privilege**
      ([`aieng.agents.tools`](../../src/aieng/agents/tools.py))
- [ ] Typed, constrained parameters — enums wherever possible
- [ ] Human confirmation on every consequential action, failing closed
- [ ] At least one knowledge tool, one compute tool, and one write tool

**Memory**
- [ ] [`MemoryStore`](../../src/aieng/agents/memory.py) with supersession, recency
      weighting, and forgetting
- [ ] **Reflection**: extract durable memories from a session rather than storing
      transcripts
- [ ] A user-facing "what do you remember about me?" view with per-item deletion

**Observability and evaluation**
- [ ] A complete `RunTrace` per run: agent version, model, every tool call and
      result, tokens, cost, latency
- [ ] **20 reference trajectories** — the ideal tool sequence, written by hand
- [ ] Scored with [`aieng.evals.trajectory`](../../src/aieng/evals/trajectory.py),
      reporting the **verdict distribution**, not a success rate
- [ ] Failure injection: tools that error, time out, return empty, return garbage —
      and the agent recovers from each

## The two things that make this real

**1. Reference trajectories.** Write out, by hand, the ideal sequence of tool calls
for twenty tasks. Then compare what your agent actually did.

Twenty of these teach you more than five hundred end-to-end scores, because they
are the only way to see **efficiency failures** — the correct answer reached
through fourteen calls when two would do. No accuracy metric shows you that.

**2. Red-teaming your own agent.** Put a document containing injected instructions
somewhere your agent will retrieve it, and try to make it call a tool it should
not. Write down what worked.

Then add least-privilege scoping and confirm the same attack now fails
*structurally* — because the capability is absent, not because the model resisted
persuasion. That distinction is the whole security model
([H5](../../books/04-ai-engineering-huyen/notes/ch05.md),
[L5](../../books/05-ai-agents-in-action-lanham/notes/ch05.md)).

## Pitfalls

- **No step cap or cost budget.** The classic agent incident, and the classic
  surprise invoice.
- **Raising on tool errors.** Recovery is the behaviour you are building.
- **Aborting on a detected loop instead of informing the agent.** Tell it what it
  repeated; it can usually recover.
- **Broad tool permissions "for flexibility."** That is the vulnerability, phrased
  as a convenience.
- **Trusting retrieved content.** It came through your pipeline and is still
  attacker-controllable.
- **Evaluating only the final answer.** You will never see efficiency or safety
  failures.
- **A single run per task.** Agents are non-deterministic; run each several times.
- **More than ~12 tools.** Selection accuracy degrades measurably.

## Stretch

- Plan-then-execute with **validation before execution**, compared against the
  reactive loop on the same tasks.
- **Replanning** with explicit triggers, a cap, and preserved findings. Then remove
  the preserved findings and watch it replan into the same failure
  ([L11](../../books/05-ai-agents-in-action-lanham/notes/ch11.md)).
- Escalation-on-uncertainty using logprobs as the confidence signal.
- Sandboxed code execution: container, no network, timeout, memory limit.
- A critic agent with an **explicit rubric**; measure quality gain per round and
  find where it flattens (usually round two).
- Close the loop: failed runs become eval cases, corrections become memories,
  patterns become prompt revisions.

## Getting started

```bash
pip install -e ".[agents,rag]"
python -m projects.p5_agent_platform.run "find the highest-cost step in my last 10 runs"
python -m projects.p5_agent_platform.evaluate --suite suites/trajectories.yaml
```

Give it the retrieval tool from [p2](../p2-rag-over-my-library/) and you have an
agent that can research across your whole library.
