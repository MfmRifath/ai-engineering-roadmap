# Hands-On Large Language Models

**Jay Alammar & Maarten Grootendorst · O'Reilly · 2024 · 12 chapters in 3 parts**
Roadmap phases [4](../../ROADMAP.md#phase-4--llm-intuition),
[6](../../ROADMAP.md#phase-6--fine-tuning--representation)

## Why this book, second

Jay Alammar wrote *The Illustrated Transformer*, the blog post that taught a generation of
engineers what attention is. This book is that talent applied to the whole field.

Its job in this roadmap is **intuition before implementation**. Read it before Raschka and his
code will feel like transcription of something you already understand. Read it after and you
will spend Raschka's seven chapters confused about *why*, not *how*.

The trap is that the diagrams make it feel easy. Feeling that you understand a picture is not
the same as understanding the thing. That is what Phase 5 is for.

## Structure

| Part | Chapters | What it does |
|---|---|---|
| I — Understanding Language Models | 1–3 | How LLMs work. The most valuable third. |
| II — Using Pretrained Models | 4–9 | Applying them: classification, clustering, prompting, RAG |
| III — Training & Fine-Tuning | 10–12 | Adapting them: embeddings, PEFT, LoRA |

Parts I–II are Phase 4. Part III is Phase 6, after you have built a GPT yourself.

## Chapter map

| # | Chapter | Note | Phase | Priority |
|---|---|---|---|---|
| 1 | An Introduction to Large Language Models | [ch01](notes/ch01.md) | 4 | Core |
| 2 | Tokens and Embeddings | [ch02](notes/ch02.md) | 4 | **Critical** |
| 3 | Looking Inside Large Language Models | [ch03](notes/ch03.md) | 4 | **Critical** |
| 4 | Text Classification | [ch04](notes/ch04.md) | 4 | Core |
| 5 | Text Clustering and Topic Modeling | [ch05](notes/ch05.md) | 4 | Skimmable |
| 6 | Prompt Engineering | [ch06](notes/ch06.md) | 4 | Core |
| 7 | Advanced Text Generation Techniques and Tools | [ch07](notes/ch07.md) | 4 | Core |
| 8 | Semantic Search and Retrieval-Augmented Generation | [ch08](notes/ch08.md) | 4 | **Critical** |
| 9 | Multimodal Large Language Models | [ch09](notes/ch09.md) | 4 | Context |
| 10 | Creating Text Embedding Models | [ch10](notes/ch10.md) | 6 | Core |
| 11 | Fine-Tuning Representation Models for Classification | [ch11](notes/ch11.md) | 6 | Core |
| 12 | Fine-Tuning Generation Models | [ch12](notes/ch12.md) | 6 | **Critical** |

## How to read chapter 3

Chapter 3 is the single most important chapter in the book, and it rewards a specific approach:

1. Read it once at speed, letting the diagrams wash over you. Do not stop at confusion.
2. Read it again with a pen. Draw the forward pass yourself — tokens in, logits out.
3. Read it a third time asking only: *where does the KV cache live and what does it save?*

If you can draw the diagram from memory afterwards, Phase 5 will be straightforward.

## What ages fastest

The library ecosystem. The book uses `transformers`, `sentence-transformers`, LangChain, and
specific model checkpoints; APIs churn and checkpoints get superseded. Expect to adapt code.

The **explanations** do not age. Tokenization, embeddings, the transformer forward pass,
contrastive training, and retrieval work the same way they did in 2024.

`[dated]` markers in the notes flag API-level drift.

## Where these ideas resurface

| From Alammar | Reappears as |
|---|---|
| Tokenization (ch. 2) | BPE implemented from scratch — [R2](../03-build-llm-from-scratch-raschka/notes/ch02.md) |
| The forward pass (ch. 3) | The GPT you write — [R4](../03-build-llm-from-scratch-raschka/notes/ch04.md) |
| KV caching (ch. 3) | Inference optimization — [H9](../04-ai-engineering-huyen/notes/ch09.md) |
| Sampling (ch. 6) | Generation internals — [R5](../03-build-llm-from-scratch-raschka/notes/ch05.md) |
| RAG (ch. 8) | Production RAG and its failure modes — [H6](../04-ai-engineering-huyen/notes/ch06.md) |
| Agents (ch. 7) | The whole of [Lanham](../05-ai-agents-in-action-lanham/) |
| LoRA (ch. 12) | Fine-tuning economics — [H7](../04-ai-engineering-huyen/notes/ch07.md) |

Full map: [CURRICULUM.md](../../CURRICULUM.md).

## Companion material

- Official notebooks: [github.com/HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
  — Colab-ready, which matters if you lack a GPU.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — read
  alongside chapter 3.
- [BERTopic docs](https://maartengr.github.io/BERTopic/) — Grootendorst's library, used in ch. 5.

## Project

[p2 — RAG over my library](../../projects/p2-rag-over-my-library/) builds directly on chapter 8,
using the PDFs in your gitignored `library/` as the corpus.
