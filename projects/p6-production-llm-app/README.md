# p6 — Production LLM app

**The capstone · Draws on [H10](../../books/04-ai-engineering-huyen/notes/ch10.md)
and everything before it**

One application that uses all of it: retrieval, an agent, guardrails, routing,
caching, evaluation, tracing, and a feedback loop. Built in Huyen's order, adding
each layer only when the previous one is not enough.

## Why this project

Every previous project proved one skill. This proves you can compose them — which
is the actual job. It is also the thing you can show someone.

## Pick something you will actually use

The single biggest predictor of whether this gets finished. Candidates:

- A study assistant over your library ([p2](../p2-rag-over-my-library/)) that
  quizzes you from your own flashcards
- A code-review assistant for your repositories
- A research assistant that reads papers and maintains notes
- Something for a hobby of yours that nobody has built

Not a generic chatbot. Something with a specific job you can define success for.

## Build in this order

Huyen's five steps, and the ordering is the point — **do not build the whole
architecture up front.** Add each layer when the previous one stops being enough,
and write down what forced the change.

### 1. Enhance context
Start with a model behind an API. Add retrieval, tools, and memory only where the
model demonstrably lacks what it needs. Most quality problems are context problems.

- [ ] Baseline: plain model, no retrieval. **Measure it.** You need this number.
- [ ] Add RAG. Measure the improvement.

### 2. Guardrails
- [ ] **Input**: PII detection, scope filtering, injection detection, rate limits
- [ ] **Output**: schema validation, PII leakage, refusal handling
- [ ] Measure the latency cost at p50 and p95, then decide about streaming
      deliberately

### 3. Router and gateway
- [ ] A gateway with provider abstraction, fallback chain, retries, and cost accounting
- [ ] A router: cheap model for easy queries, expensive for hard ones
- [ ] **Report the cost saving and any quality loss.** This is usually the single
      largest win available, because most traffic is easy.

### 4. Caching
- [ ] Exact-match cache
- [ ] Prefix ordering so provider-side KV caching hits — stable content first
- [ ] Semantic cache **only if** you evaluate hit *correctness*, not just hit rate.
      "Refund policy for EU customers" and "for US customers" are near-identical
      embeddings with different right answers.

### 5. Agent patterns
- [ ] Only where the workflow genuinely needs dynamic decisions
- [ ] Everything from [p5](../p5-agent-platform/): caps, budgets, tracing, recovery

## Definition of done

- [ ] It solves a real problem for you, and you have used it for a week
- [ ] Complete traces: prompt version, model version, retrieved chunks, tool calls,
      tokens, cost, latency
- [ ] An eval suite ([p4](../p4-llm-eval-harness/)) run on every prompt change
- [ ] Cost per request tracked, with an alert threshold
- [ ] **Implicit feedback** captured — regeneration, copy, abandonment
- [ ] Ten production failures converted into eval cases
- [ ] A README a stranger could deploy from
- [ ] An honest **limitations** section: what it gets wrong, and how you know

## The loop that makes it improve

```
production run → full trace → outcome (environmental or human)
   → failures become eval cases
   → corrections become memories
   → patterns become prompt revisions
   → re-run the eval suite → deploy
```

Without this an app does not get better, it just gets older. Building it is what
separates this from every other side project.

## Pitfalls

- **Building all five layers before measuring the first.** Premature routers and
  caches are complexity with no payoff.
- **No baseline.** You cannot claim RAG helped without the number from before it.
- **Optimizing directly for thumbs-up.** Users prefer confident and agreeable over
  correct.
- **Incomplete traces.** Without prompt and model version you cannot attribute a
  regression.
- **No fallback chain.** Providers have outages; yours should not.
- **No cost alerting.** Runaway loops are discovered by invoice.
- **Ignoring implicit feedback.** It is higher-volume and less biased than ratings.

## Stretch

- Shadow-deploy a candidate prompt against real traffic; compare offline.
- A/B two models with statistical significance on your metric.
- A dashboard: quality, cost, latency, refusal rate over time.
- Publish it. A working, evaluated, documented LLM app is a stronger portfolio
  piece than any certificate.

## Getting started

```bash
pip install -e ".[all]"
```

Then open [H10](../../books/04-ai-engineering-huyen/notes/ch10.md) and start at
step 1 — with the baseline measurement, not the architecture.
