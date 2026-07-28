# AI Engineering: Building Applications with Foundation Models

**Chip Huyen · O'Reilly · 2025 · 10 chapters**
Roadmap phases [6](../../ROADMAP.md#phase-6--fine-tuning--representation),
[7](../../ROADMAP.md#phase-7--production-ai-engineering)

## Why this book, fourth

This is the book that turns you from someone who can build a demo into someone who can ship.

Raschka teaches you what the model *is*. Huyen teaches you what happens when you put it in front
of users: it is too slow, it costs too much, it is wrong in ways you cannot detect, and nobody
can tell you whether last week's prompt change helped or hurt.

Chapters 3 and 4 — **evaluation** — are the heart of it, and the most under-practiced skill in
the field. Almost everyone building LLM products is flying on vibes. The engineers who can put a
number on quality are the ones who get trusted with decisions.

## Chapter map

| # | Chapter | Note | Phase | Priority |
|---|---|---|---|---|
| 1 | Introduction to Building AI Applications with Foundation Models | [ch01](notes/ch01.md) | 7 | Skimmable |
| 2 | Understanding Foundation Models | [ch02](notes/ch02.md) | 7 | Core |
| 3 | Evaluation Methodology | [ch03](notes/ch03.md) | 7 | **Critical** |
| 4 | Evaluate AI Systems | [ch04](notes/ch04.md) | 7 | **Critical** |
| 5 | Prompt Engineering | [ch05](notes/ch05.md) | 7 | Core |
| 6 | RAG and Agents | [ch06](notes/ch06.md) | 7 | **Critical** |
| 7 | Finetuning | [ch07](notes/ch07.md) | 6 | **Critical** |
| 8 | Dataset Engineering | [ch08](notes/ch08.md) | 6 | Core |
| 9 | Inference Optimization | [ch09](notes/ch09.md) | 7 | Core |
| 10 | AI Engineering Architecture and User Feedback | [ch10](notes/ch10.md) | 7 | **Critical** |

Chapters 7–8 are pulled forward into Phase 6 so the fine-tuning material lands together with
Alammar's PEFT chapters, while you still have Raschka's training loop fresh.

## Give evaluation double the time

Whatever you budget for chapters 3 and 4, double it. The reason is structural: **open-ended
output has no ground truth**, so every technique you learned in Géron chapter 3 stops applying
directly. Huyen's answer is a layered one, and each layer has a distinct failure mode:

| Layer | Technique | Fails when |
|---|---|---|
| Exact | Functional correctness, string match | The task has many valid answers |
| Reference | Similarity to a gold answer | Good answers are phrased differently |
| Model | AI as a judge | The judge is biased, inconsistent, or self-preferring |
| Comparative | Pairwise ranking, Elo | You need an absolute quality bar, not a ranking |

The lesson is not "use AI as a judge." It is that you compose these, and you evaluate your
evaluator. [p4](../../projects/p4-llm-eval-harness/) makes you build it.

## The two decision frameworks worth memorizing

Two sections earn their keep long after you have forgotten the rest:

- **Chapter 7, "When to Finetune" / "Reasons Not to Finetune."** The default answer is
  *don't* — try prompting, then RAG, then fine-tuning, in that order. Huyen gives you the
  argument, not just the conclusion.
- **Chapter 10, the five-step architecture.** Enhance context → add guardrails → add a router
  and gateway → add caching → add agent patterns. Every production LLM system converges on some
  version of this. Knowing the order saves you a rewrite.

## What ages fastest

Specific model names, benchmark scores, and prices. Written in 2025, and the frontier moves
monthly. **The methodology does not move.** How to *decide* which model, how to *measure* a
regression, how to budget memory — those hold regardless of which model is on top this quarter.

When a note cites a specific model or price, it is marked `[dated]` and treated as an
illustration of the method, not a recommendation.

## Where these ideas resurface

| From Huyen | Connects to |
|---|---|
| Sampling & structured output (ch. 2) | Implemented in [R5](../03-build-llm-from-scratch-raschka/notes/ch05.md) |
| Perplexity & entropy (ch. 3) | The loss you trained in [R5](../03-build-llm-from-scratch-raschka/notes/ch05.md) |
| Evaluation criteria (ch. 4) | Precision/recall from [G3](../01-hands-on-ml-geron/notes/ch03.md) |
| Prompt injection (ch. 5) | Tool-use safety in [L5](../05-ai-agents-in-action-lanham/notes/ch05.md) |
| RAG (ch. 6) | The naive version from [A8](../02-hands-on-llms-alammar/notes/ch08.md) |
| Agents (ch. 6) | Expanded across all of [Lanham](../05-ai-agents-in-action-lanham/) |
| LoRA & memory math (ch. 7) | [R App. E](../03-build-llm-from-scratch-raschka/notes/ch07.md), [A12](../02-hands-on-llms-alammar/notes/ch12.md) |
| KV cache & batching (ch. 9) | The cache you saw in [A3](../02-hands-on-llms-alammar/notes/ch03.md) |
| Serving architecture (ch. 10) | Scaled up from [G19](../01-hands-on-ml-geron/notes/ch19.md) |

Full map: [CURRICULUM.md](../../CURRICULUM.md).

## Companion material

- [huyenchip.com/blog](https://huyenchip.com/blog/) — much of the book's thinking appeared here
  first, and the posts stay updated.
- *Designing Machine Learning Systems* (Huyen, 2022) — the sibling volume, for classical ML
  systems. Read it if you work on ML platforms.

## Projects

- [p4 — LLM eval harness](../../projects/p4-llm-eval-harness/) — chapters 3–4 made executable.
- [p6 — production LLM app](../../projects/p6-production-llm-app/) — the capstone, structured
  around chapter 10's five-step architecture.
