# Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow

**Aurélien Géron · O'Reilly · 2nd edition, 2019 · 19 chapters**
Roadmap phases [1](../../ROADMAP.md#phase-1--ml-foundations),
[2](../../ROADMAP.md#phase-2--deep-learning-foundations),
[3](../../ROADMAP.md#phase-3--sequences-attention--generative-precursors)

## Why this book, first

Every LLM skill you are here for rests on habits this book installs: hold out a test set and
never touch it, distrust a single metric, expect your model to be worse than your validation
score says. People who skip straight to transformers can call an API. They cannot tell you
whether their RAG system got better or their eval set got easier.

Géron is also simply the best-written ML textbook in print. The exercises are worth doing.

## Read it in three passes

| Pass | Chapters | Phase | Why |
|---|---|---|---|
| 1 | 1–9 | Phase 1 | The core. Read slowly, do the exercises. |
| 2 | 10–15, 19 | Phase 2 | Deep learning. Read for **concepts**, not APIs. |
| 3 | 16–18 | Phase 3 | The bridge to transformers, plus context for RLHF. |

Chapter 19 is pulled forward into pass 2 because deployment belongs with training, not at the end.

## A word on the edition

This is the **2nd edition (2019)**. A 3rd edition (2022) exists and is better if you are buying
new — it uses Keras 3 idioms and adds transformer coverage.

What that means for you:

- **Chapters 1–9 are timeless.** Scikit-learn's API has barely moved. Read as written.
- **Chapters 10–15, 19 have aged in the details.** TF 2.x-era APIs, `tf.data`, TF Serving. The
  *concepts* — batch norm, dropout, learning-rate schedules, input pipelines — are exactly right.
  You will write PyTorch from Phase 5 onward, so read for the idea and skim the code.
- **Chapter 16 predates the LLM era.** It ends where the modern story begins. That is precisely
  what makes it useful: you feel the problem before you are handed the solution.

Notes flag stale API details as **`[dated]`** where it matters.

## Chapter map

| # | Chapter | Note | Phase | Priority |
|---|---|---|---|---|
| 1 | The Machine Learning Landscape | [ch01](notes/ch01.md) | 1 | Core |
| 2 | End-to-End Machine Learning Project | [ch02](notes/ch02.md) | 1 | **Critical** |
| 3 | Classification | [ch03](notes/ch03.md) | 1 | **Critical** |
| 4 | Training Models | [ch04](notes/ch04.md) | 1 | **Critical** |
| 5 | Support Vector Machines | [ch05](notes/ch05.md) | 1 | Skimmable |
| 6 | Decision Trees | [ch06](notes/ch06.md) | 1 | Core |
| 7 | Ensemble Learning and Random Forests | [ch07](notes/ch07.md) | 1 | Core |
| 8 | Dimensionality Reduction | [ch08](notes/ch08.md) | 1 | Core |
| 9 | Unsupervised Learning Techniques | [ch09](notes/ch09.md) | 1 | Core |
| 10 | Introduction to ANNs with Keras | [ch10](notes/ch10.md) | 2 | Core |
| 11 | Training Deep Neural Networks | [ch11](notes/ch11.md) | 2 | **Critical** |
| 12 | Custom Models and Training with TensorFlow | [ch12](notes/ch12.md) | 2 | Skimmable |
| 13 | Loading and Preprocessing Data with TensorFlow | [ch13](notes/ch13.md) | 2 | Core |
| 14 | Deep Computer Vision Using CNNs | [ch14](notes/ch14.md) | 2 | Core |
| 15 | Processing Sequences Using RNNs and CNNs | [ch15](notes/ch15.md) | 2 | Core |
| 16 | NLP with RNNs and Attention | [ch16](notes/ch16.md) | 3 | **Critical** |
| 17 | Autoencoders and GANs | [ch17](notes/ch17.md) | 3 | Context |
| 18 | Reinforcement Learning | [ch18](notes/ch18.md) | 3 | Context |
| 19 | Training and Deploying at Scale | [ch19](notes/ch19.md) | 2 | Core |

**Critical** — do the exercises. **Core** — read fully, note it. **Skimmable** — understand the
idea, move on. **Context** — background you will be glad of later.

## If you are short on time

The irreducible minimum is **2, 3, 4, 11, 16**. Chapter 2 gives you the workflow, 3 gives you
evaluation, 4 gives you optimization, 11 gives you the practical training toolkit, and 16 sets
up the transformer. Everything else can be looked up.

Do not skip 3. Precision/recall confusion is the single most common failure in applied ML, and
it reappears verbatim in LLM evaluation ([Huyen ch. 4](../04-ai-engineering-huyen/notes/ch04.md)).

## Where these ideas resurface

| From Géron | Reappears as |
|---|---|
| Test-set discipline (ch. 2) | Eval set contamination — [H4](../04-ai-engineering-huyen/notes/ch04.md) |
| Precision/recall (ch. 3) | Retrieval quality in RAG — [H6](../04-ai-engineering-huyen/notes/ch06.md) |
| Regularization (ch. 4, 11) | Overfitting during fine-tuning — [H7](../04-ai-engineering-huyen/notes/ch07.md) |
| Transfer learning (ch. 11, 14) | PEFT and LoRA — [A12](../02-hands-on-llms-alammar/notes/ch12.md) |
| Embeddings (ch. 13, 17) | Token & text embeddings — [A2](../02-hands-on-llms-alammar/notes/ch02.md) |
| Attention (ch. 16) | The whole transformer — [R3](../03-build-llm-from-scratch-raschka/notes/ch03.md) |
| Policy gradients (ch. 18) | RLHF and preference tuning — [H2](../04-ai-engineering-huyen/notes/ch02.md) |
| Serving (ch. 19) | Inference optimization — [H9](../04-ai-engineering-huyen/notes/ch09.md) |

Full map: [CURRICULUM.md](../../CURRICULUM.md).

## Companion material

- Official notebooks: [github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2)
  (3rd ed: [handson-ml3](https://github.com/ageron/handson-ml3))
- Solutions to every exercise are in the book's appendix — write your answer first.

## Project

[p1 — end-to-end ML service](../../projects/p1-end-to-end-ml-service/) is chapter 2 taken all
the way to a served, containerized endpoint.
