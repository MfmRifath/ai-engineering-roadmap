# AI Agents in Action

**Michael Lanham · Manning · 2025 · 11 chapters**
Roadmap phase [8](../../ROADMAP.md#phase-8--agents)

## Why this book, last

Agents are the least settled area in the field, which is exactly why it comes after the other
four. Build an agent before you understand evaluation and you will produce something that
demos beautifully and fails silently — the worst possible outcome, because you will not know.

By the time you get here you should be able to ask the hard questions of every pattern this
book shows you: *how would I know if this worked? what does it cost per call? what happens when
the third tool call returns garbage?*

## Read this one critically

More than any other book on the list, this one will age. It leans on specific frameworks,
specific APIs, and a specific moment in the agent tooling cycle. That is not a flaw — a book
about agents has to be concrete — but it changes how you read it.

**Separate the durable from the disposable:**

| Durable — will still be true in 2030 | Disposable — will churn within a year |
|---|---|
| Agents = LLM + actions + memory + planning + a loop | Which framework wraps that loop |
| Tool schemas need clear names and typed args | The exact function-calling JSON dialect |
| Memory is retrieval with a write path | Which vector store you use |
| Planning fails; you need recovery, not just retry | The specific planner prompt |
| Trajectory evaluation ≠ output evaluation | The eval library |
| Multi-agent = a coordination problem | The orchestration SDK |

The notes here foreground the left column and mark the right `[framework-specific]`.

## Chapter map

| # | Chapter | Note | Priority |
|---|---|---|---|
| 1 | Introduction to Agents and Their World | [ch01](notes/ch01.md) | Core |
| 2 | Harnessing the Power of Large Language Models | [ch02](notes/ch02.md) | Skimmable |
| 3 | Engaging GPT Assistants | [ch03](notes/ch03.md) | Skimmable |
| 4 | Exploring Multi-Agent Systems | [ch04](notes/ch04.md) | Core |
| 5 | Empowering Agents with Actions | [ch05](notes/ch05.md) | **Critical** |
| 6 | Building Autonomous Assistants | [ch06](notes/ch06.md) | Core |
| 7 | Assembling and Using an Agent Platform | [ch07](notes/ch07.md) | Core |
| 8 | Understanding Agent Memory and Knowledge | [ch08](notes/ch08.md) | **Critical** |
| 9 | Mastering Agent Prompts with Prompt Flow | [ch09](notes/ch09.md) | Core |
| 10 | Agent Reasoning and Evaluation | [ch10](notes/ch10.md) | **Critical** |
| 11 | Agent Planning and Feedback | [ch11](notes/ch11.md) | **Critical** |

Chapters 2–3 overlap heavily with material you have already covered in
[Alammar](../02-hands-on-llms-alammar/) and [Huyen](../04-ai-engineering-huyen/). Skim them,
note the deltas, move on.

## The four chapters that matter most

If you read only four: **5, 8, 10, 11.**

- **5 — Actions.** Tool use is the thing that makes an agent an agent rather than a chatbot.
  Everything else is decoration on top of a good action layer. This is also where the security
  surface lives — pair it with [Huyen ch. 5](../04-ai-engineering-huyen/notes/ch05.md) on
  prompt injection, because a tool-using agent with an injection vulnerability is a remote code
  execution bug wearing a friendly interface.
- **8 — Memory.** Once you have read [A8](../02-hands-on-llms-alammar/notes/ch08.md) and
  [H6](../04-ai-engineering-huyen/notes/ch06.md), you will recognize agent memory as RAG with a
  write path. That reframing makes the whole chapter click.
- **10 — Reasoning and evaluation.** Evaluating a *trajectory* is a genuinely different problem
  from evaluating an output. A correct answer reached through six wasteful tool calls is a
  failure you must be able to see.
- **11 — Planning and feedback.** Where agents actually break. Plans go stale mid-execution;
  the interesting engineering is replanning, not planning.

## The honest state of agents

Worth holding in mind while you read, and worth writing in your own notes:

- Reliability compounds downward. Ten sequential steps at 95% each is 60% end to end.
- Most production "agents" are constrained workflows with one or two LLM decision points, not
  open-ended loops. This is a feature.
- The failure mode is rarely a wrong answer. It is a loop, a runaway cost, or a confidently
  wrong tool call.
- Multi-agent systems multiply cost and latency and are frequently a way of avoiding a clear
  problem decomposition. Sometimes they are genuinely right. Ask which.

None of that means don't build agents. It means build them with the evaluation discipline from
Phase 7 already in hand.

## Where these ideas connect

| From Lanham | Builds on |
|---|---|
| Agent definition (ch. 1) | [H6](../04-ai-engineering-huyen/notes/ch06.md) agents section |
| Tool use (ch. 5) | Structured output — [H2](../04-ai-engineering-huyen/notes/ch02.md) |
| Multi-agent (ch. 4, 7) | Prompt chaining — [A7](../02-hands-on-llms-alammar/notes/ch07.md) |
| Memory (ch. 8) | RAG — [A8](../02-hands-on-llms-alammar/notes/ch08.md), [H6](../04-ai-engineering-huyen/notes/ch06.md) |
| Prompt flow (ch. 9) | Prompt versioning — [H5](../04-ai-engineering-huyen/notes/ch05.md) |
| Reasoning (ch. 10) | Chain-of-thought — [A6](../02-hands-on-llms-alammar/notes/ch06.md) |
| Evaluation (ch. 10) | Eval methodology — [H3](../04-ai-engineering-huyen/notes/ch03.md), [H4](../04-ai-engineering-huyen/notes/ch04.md) |
| Feedback loops (ch. 11) | User feedback — [H10](../04-ai-engineering-huyen/notes/ch10.md) |
| Planning & rewards (ch. 11) | RL foundations — [G18](../01-hands-on-ml-geron/notes/ch18.md) |

Full map: [CURRICULUM.md](../../CURRICULUM.md).

## Companion material

- [Manning book page](https://www.manning.com/books/ai-agents-in-action) — source code is linked
  from there.
- Worth reading alongside, as durable primary sources:
  [ReAct](https://arxiv.org/abs/2210.03629) ·
  [Toolformer](https://arxiv.org/abs/2302.04761) ·
  [Reflexion](https://arxiv.org/abs/2303.11366) ·
  [Anthropic — Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

## Projects

- [p5 — agent platform](../../projects/p5-agent-platform/) — chapters 5–11 assembled: tools,
  memory, planning, and a loop that recovers.
- [p6 — production LLM app](../../projects/p6-production-llm-app/) — the capstone.
