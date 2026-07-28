# p4 — LLM eval harness

**After Phase 7 · Draws on [H3](../../books/04-ai-engineering-huyen/notes/ch03.md),
[H4](../../books/04-ai-engineering-huyen/notes/ch04.md)**

A CLI that scores models and prompts against your own evaluation set, and tells
you when a change made things worse.

## Why this project

Evaluation is the skill that separates people who ship AI from people who demo it,
and almost nobody builds this. Having your own harness means you can answer
"did that prompt change help?" with a number instead of an opinion — and that
single capability will make you more useful than knowing any particular technique.

## Spec

```bash
# score a model on a suite
aieval run --suite suites/summarization.yaml --model claude-sonnet-4-5

# compare two variants and report the delta with significance
aieval compare --suite suites/summarization.yaml \
               --baseline prompts/v3.md --candidate prompts/v4.md

# fail CI if quality regressed
aieval check --suite suites/summarization.yaml --against baselines/v3.json --threshold 0.02
```

## Definition of done

- [ ] Suites defined in files (YAML/JSON), version-controlled alongside prompts
- [ ] **An evaluation guideline** per suite: what a response must contain, what
      disqualifies it, and worked examples of good/borderline/bad. Writing this
      is most of the work and most of the value.
- [ ] At least three scorer types:
  - **Functional/exact** — the gold standard where it applies
  - **Similarity** — against reference answers
  - **LLM judge** — using the guideline as its rubric
- [ ] **Judge biases controlled**: positions randomized or both orders evaluated,
      and length recorded so you can detect verbosity bias
- [ ] **Results sliced** — by category, difficulty, and input length. An aggregate
      hides exactly the failures you need.
- [ ] Cost and latency (p50/p95) reported alongside quality, per
      [H4](../../books/04-ai-engineering-huyen/notes/ch04.md)
- [ ] Results persisted so runs are comparable over time
- [ ] `check` exits non-zero on regression, so it can gate a deploy

## The step nobody does

**Validate your judge.** Hand-rate 30 outputs yourself, then correlate against the
judge's scores. Report the agreement.

If agreement is poor, your judge is a random number generator with good
handwriting and every decision you make with it is noise. This one measurement
is what separates an eval harness from eval theatre.

## Pitfalls

- **Absolute 0–100 judge scores.** Poorly calibrated and inconsistent. Prefer
  pairwise comparison, or a rubric with anchored examples.
- **Not controlling position.** Judges favour whichever answer comes first.
- **Ignoring verbosity bias.** Longer answers score higher. Record lengths.
- **An eval set too small for the effect you care about.** 50 examples cannot
  resolve a 2% difference. Compute what you need.
- **Only success cases.** Include inputs that *should* be refused, and check the
  model refuses rather than inventing.
- **A static eval set.** Every production failure becomes a case, or you will fix
  things twice.
- **Reporting one number.** Slice it.

## Stretch

- Bootstrap confidence intervals on the score difference, so "better" is a claim
  you can defend.
- An ensemble of judges from different model families; report disagreement rate.
- HTML report with per-example diffs between baseline and candidate.
- A GitHub Action that runs the suite on prompt changes and comments the delta.
- Track judge–human agreement over time; it drifts as models change.
- Reuse [`aieng.evals.trajectory`](../../src/aieng/evals/trajectory.py) so the same
  harness can score agent runs for [p5](../p5-agent-platform/).

## Getting started

```bash
pip install -e ".[dev,agents]"
python -m projects.p4_eval_harness.cli run --suite suites/example.yaml --model <model>
```

Start with **one** suite of 20 examples for a task you actually care about. A
small harness you use beats a comprehensive one you do not.
