# Build a Large Language Model (From Scratch)

**Sebastian Raschka · Manning · 2024 · 7 chapters + appendices**
Roadmap phase [5](../../ROADMAP.md#phase-5--transformers-from-scratch)

## Why this book is the centre of the roadmap

Seven chapters. Raw text in chapter 2, a GPT you trained yourself by chapter 5, fine-tuned twice
by chapter 7. No `AutoModel`, no hand-waving — you write the tokenizer, the attention, the
transformer block, the training loop.

After this, "the model hallucinated" stops being a mystical statement and becomes a claim about
a softmax over a vocabulary. That shift is the whole point. Every downstream skill — evaluation,
fine-tuning decisions, inference optimization, debugging an agent — gets easier because you know
what is actually happening.

**If you only read one of the five books, read this one.**

## How to read it

**Type every line.** Do not clone the repo and run it. The muscle memory of writing
`attn_scores = queries @ keys.transpose(-2, -1)` and getting the shapes wrong is the learning.

Raschka's teaching method in chapter 3 is worth naming, because it is unusual: he builds
attention **four times**, each version fixing a limitation of the last.

1. Simplified self-attention, no trainable weights — just dot products and softmax
2. Self-attention with trainable W_q, W_k, W_v
3. Causal attention — add the mask, add dropout
4. Multi-head attention — stack it, then batch it efficiently

Do not skip to version 4. The progression *is* the explanation.

## Chapter map

| # | Chapter | Note | You end up with | Priority |
|---|---|---|---|---|
| 1 | Understanding Large Language Models | [ch01](notes/ch01.md) | The mental model and the plan | Core |
| 2 | Working with Text Data | [ch02](notes/ch02.md) | A BPE tokenizer, a sliding-window dataset, embeddings | **Critical** |
| 3 | Coding Attention Mechanisms | [ch03](notes/ch03.md) | Causal multi-head attention, from scratch | **Critical** |
| 4 | Implementing a GPT Model from Scratch | [ch04](notes/ch04.md) | A 124M-parameter GPT-2 that generates (badly) | **Critical** |
| 5 | Pretraining on Unlabeled Data | [ch05](notes/ch05.md) | A trained model + loaded GPT-2 weights | **Critical** |
| 6 | Fine-tuning for Classification | [ch06](notes/ch06.md) | A spam classifier built on your own GPT | Core |
| 7 | Fine-tuning to Follow Instructions | [ch07](notes/ch07.md) | An instruction-following model | Core |

### Appendices — do not skip these

| | Appendix | Why it matters |
|---|---|---|
| A | Introduction to PyTorch | Skip only if you are already fluent. Otherwise read it first. |
| C | Exercise solutions | Attempt first, then check. |
| D | Bells and whistles for the training loop | LR warmup, cosine decay, gradient clipping — the difference between a loop that works and one that trains. |
| E | Parameter-efficient fine-tuning with LoRA | **~30 pages that unlock the entire modern fine-tuning stack.** Pairs with [H7](../04-ai-engineering-huyen/notes/ch07.md). |

## Hardware

You do not need a GPU. The book is designed for a laptop:

- **Chapters 2–4** — CPU is fine. You are building, not training.
- **Chapter 5** — pretraining on a small corpus takes minutes on CPU, seconds on GPU. The
  chapter then loads OpenAI's released GPT-2 weights into *your* architecture, which is the most
  satisfying moment in the book: proof your implementation is correct.
- **Chapters 6–7** — fine-tuning a 124M model is laptop-scale. Larger variants want a GPU;
  Colab's free tier is enough.

## The test that proves you got it right

Your attention implementation should match PyTorch's to floating-point tolerance:

```bash
pytest tests/test_attention.py -v
```

See [`src/aieng/transformer/attention.py`](../../src/aieng/transformer/attention.py). If that
passes, chapter 3 is genuinely done.

## Where these ideas resurface

| From Raschka | Reappears as |
|---|---|
| BPE (ch. 2) | Tokenizer economics, context cost — [H2](../04-ai-engineering-huyen/notes/ch02.md) |
| Attention (ch. 3) | FlashAttention, GQA, KV cache — [H9](../04-ai-engineering-huyen/notes/ch09.md) |
| The GPT block (ch. 4) | Architecture tradeoffs at scale — [H2](../04-ai-engineering-huyen/notes/ch02.md) |
| Sampling & temperature (ch. 5) | Controlling output — [A6](../02-hands-on-llms-alammar/notes/ch06.md) |
| Loss & perplexity (ch. 5) | Evaluation metrics — [H3](../04-ai-engineering-huyen/notes/ch03.md) |
| Instruction tuning (ch. 7) | Post-training and dataset design — [H8](../04-ai-engineering-huyen/notes/ch08.md) |
| LoRA (App. E) | The fine-tuning decision — [H7](../04-ai-engineering-huyen/notes/ch07.md) |

Full map: [CURRICULUM.md](../../CURRICULUM.md).

## Companion material

- Official code: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  — extremely well maintained, with bonus material beyond the book. Use it to check your work,
  not to skip it.
- Raschka's [Ahead of AI](https://magazine.sebastianraschka.com/) newsletter.
- Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the same journey
  in video form. Excellent second pass.

## Project

[p3 — nanoGPT from scratch](../../projects/p3-nanogpt-from-scratch/) is chapters 2–5 assembled
into one trainable repo, with your own corpus.
