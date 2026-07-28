# The Roadmap

Eight phases, 59 chapters, five books, six projects. Roughly **30 weeks at 8–10 hours a week**,
though the honest answer is that it takes as long as it takes.

Tick a box when you have (a) read the chapter, (b) written the note, and (c) done at least one
exercise. Then run `make progress` — [PROGRESS.md](PROGRESS.md) and the README bar update
themselves.

**Book keys:** `G` Géron · `A` Alammar · `R` Raschka · `H` Huyen · `L` Lanham

---

## How the phases fit together

```
Phase 1  ML Foundations            ──┐
Phase 2  Deep Learning             ──┼─→ you can train and evaluate a model
Phase 3  Sequences & Attention     ──┘
                 │
Phase 4  LLM Intuition             ──┐
Phase 5  Transformers From Scratch ──┼─→ you understand what an LLM is doing
Phase 6  Fine-Tuning               ──┘
                 │
Phase 7  Production AI Engineering ──┐
Phase 8  Agents                    ──┴─→ you can ship and operate one
```

Phases 1–3 are the tax you pay once. Phases 4–6 are the interesting part. Phases 7–8 are what
people will actually pay you for.

---

## Phase 1 — ML Foundations
**Weeks 1–4 · Géron 1–9 · Core**

The vocabulary and the discipline. Nothing here is about LLMs, and all of it is load-bearing:
if you cannot reason about overfitting, regularization, and a leaked test set, you cannot
evaluate an LLM system either.

- [ ] **G1** [The Machine Learning Landscape](books/01-hands-on-ml-geron/notes/ch01.md)
- [ ] **G2** [End-to-End Machine Learning Project](books/01-hands-on-ml-geron/notes/ch02.md)
- [ ] **G3** [Classification](books/01-hands-on-ml-geron/notes/ch03.md)
- [ ] **G4** [Training Models](books/01-hands-on-ml-geron/notes/ch04.md)
- [ ] **G5** [Support Vector Machines](books/01-hands-on-ml-geron/notes/ch05.md)
- [ ] **G6** [Decision Trees](books/01-hands-on-ml-geron/notes/ch06.md)
- [ ] **G7** [Ensemble Learning and Random Forests](books/01-hands-on-ml-geron/notes/ch07.md)
- [ ] **G8** [Dimensionality Reduction](books/01-hands-on-ml-geron/notes/ch08.md)
- [ ] **G9** [Unsupervised Learning Techniques](books/01-hands-on-ml-geron/notes/ch09.md)

**Prove it →** [p1 — end-to-end ML service](projects/p1-end-to-end-ml-service/)

**Exit criteria.** You can take a raw tabular dataset to a served prediction without help; you
can explain precision/recall to a product manager; you know why you never fit a scaler on the
full dataset.

---

## Phase 2 — Deep Learning Foundations
**Weeks 5–8 · Géron 10–15, 19 · Core**

Backprop, optimizers, regularization, and the input pipeline. Read for concepts — the TF/Keras
APIs in the 2nd edition have moved on, and you will write PyTorch from Phase 5 onward.

- [ ] **G10** [Introduction to Artificial Neural Networks with Keras](books/01-hands-on-ml-geron/notes/ch10.md)
- [ ] **G11** [Training Deep Neural Networks](books/01-hands-on-ml-geron/notes/ch11.md)
- [ ] **G12** [Custom Models and Training with TensorFlow](books/01-hands-on-ml-geron/notes/ch12.md)
- [ ] **G13** [Loading and Preprocessing Data with TensorFlow](books/01-hands-on-ml-geron/notes/ch13.md)
- [ ] **G14** [Deep Computer Vision Using Convolutional Neural Networks](books/01-hands-on-ml-geron/notes/ch14.md)
- [ ] **G15** [Processing Sequences Using RNNs and CNNs](books/01-hands-on-ml-geron/notes/ch15.md)
- [ ] **G19** [Training and Deploying TensorFlow Models at Scale](books/01-hands-on-ml-geron/notes/ch19.md)

**Exit criteria.** You can write a training loop from memory, diagnose a loss curve, and name
three things to try when a network will not converge.

---

## Phase 3 — Sequences, Attention & Generative Precursors
**Weeks 9–10 · Géron 16–18 · Context**

The bridge. Chapter 16 shows you the problem attention was invented to solve, which is the best
possible preparation for Phase 5. Chapters 17–18 are context rather than prerequisite — 17
underpins embeddings and diffusion, 18 underpins RLHF.

- [ ] **G16** [Natural Language Processing with RNNs and Attention](books/01-hands-on-ml-geron/notes/ch16.md)
- [ ] **G17** [Representation Learning and Generative Learning Using Autoencoders and GANs](books/01-hands-on-ml-geron/notes/ch17.md)
- [ ] **G18** [Reinforcement Learning](books/01-hands-on-ml-geron/notes/ch18.md)

**Exit criteria.** You can explain why RNNs struggle with long context and what attention does
about it — before anyone shows you the transformer diagram.

---

## Phase 4 — LLM Intuition
**Weeks 11–14 · Alammar 1–9 · Core**

Now the fun starts. This phase is about *seeing* what LLMs do. Do not skip the visual chapters
because they look easy — Alammar's diagrams are the mental model you will use for years.

- [ ] **A1** [An Introduction to Large Language Models](books/02-hands-on-llms-alammar/notes/ch01.md)
- [ ] **A2** [Tokens and Embeddings](books/02-hands-on-llms-alammar/notes/ch02.md)
- [ ] **A3** [Looking Inside Large Language Models](books/02-hands-on-llms-alammar/notes/ch03.md)
- [ ] **A4** [Text Classification](books/02-hands-on-llms-alammar/notes/ch04.md)
- [ ] **A5** [Text Clustering and Topic Modeling](books/02-hands-on-llms-alammar/notes/ch05.md)
- [ ] **A6** [Prompt Engineering](books/02-hands-on-llms-alammar/notes/ch06.md)
- [ ] **A7** [Advanced Text Generation Techniques and Tools](books/02-hands-on-llms-alammar/notes/ch07.md)
- [ ] **A8** [Semantic Search and Retrieval-Augmented Generation](books/02-hands-on-llms-alammar/notes/ch08.md)
- [ ] **A9** [Multimodal Large Language Models](books/02-hands-on-llms-alammar/notes/ch09.md)

**Prove it →** [p2 — RAG over my library](projects/p2-rag-over-my-library/)

**Exit criteria.** You can draw the forward pass of a decoder-only transformer on a whiteboard
and explain where the KV cache goes.

---

## Phase 5 — Transformers From Scratch
**Weeks 15–18 · Raschka 1–7 · Core**

The most important phase. You will implement tokenization, attention, and a full GPT, then
pretrain and fine-tune it. Type every line — do not copy-paste. The point is the muscle memory.

- [ ] **R1** [Understanding Large Language Models](books/03-build-llm-from-scratch-raschka/notes/ch01.md)
- [ ] **R2** [Working with Text Data](books/03-build-llm-from-scratch-raschka/notes/ch02.md)
- [ ] **R3** [Coding Attention Mechanisms](books/03-build-llm-from-scratch-raschka/notes/ch03.md)
- [ ] **R4** [Implementing a GPT Model from Scratch to Generate Text](books/03-build-llm-from-scratch-raschka/notes/ch04.md)
- [ ] **R5** [Pretraining on Unlabeled Data](books/03-build-llm-from-scratch-raschka/notes/ch05.md)
- [ ] **R6** [Fine-tuning for Classification](books/03-build-llm-from-scratch-raschka/notes/ch06.md)
- [ ] **R7** [Fine-tuning to Follow Instructions](books/03-build-llm-from-scratch-raschka/notes/ch07.md)

**Prove it →** [p3 — nanoGPT from scratch](projects/p3-nanogpt-from-scratch/)

**Exit criteria.** Your own GPT implementation generates coherent text, and your attention
module matches `torch.nn.MultiheadAttention` numerically. `pytest tests/test_attention.py`.

---

## Phase 6 — Fine-Tuning & Representation
**Weeks 19–21 · Alammar 10–12, Huyen 7–8 · Core**

When to adapt a model rather than prompt it, and how to do it without a GPU cluster: LoRA,
quantization, and — the part everyone underestimates — building the dataset.

- [ ] **A10** [Creating Text Embedding Models](books/02-hands-on-llms-alammar/notes/ch10.md)
- [ ] **A11** [Fine-Tuning Representation Models for Classification](books/02-hands-on-llms-alammar/notes/ch11.md)
- [ ] **A12** [Fine-Tuning Generation Models](books/02-hands-on-llms-alammar/notes/ch12.md)
- [ ] **H7** [Finetuning](books/04-ai-engineering-huyen/notes/ch07.md)
- [ ] **H8** [Dataset Engineering](books/04-ai-engineering-huyen/notes/ch08.md)

**Exit criteria.** You can compute the memory budget for fine-tuning a 7B model in your head,
and you reach for RAG before fine-tuning by default — for the right reasons.

---

## Phase 7 — Production AI Engineering
**Weeks 22–26 · Huyen 1–6, 9–10 · Core**

The professional phase. Evaluation is the skill that separates engineers from demo-builders;
chapters 3 and 4 deserve twice the time you think they do.

- [ ] **H1** [Introduction to Building AI Applications with Foundation Models](books/04-ai-engineering-huyen/notes/ch01.md)
- [ ] **H2** [Understanding Foundation Models](books/04-ai-engineering-huyen/notes/ch02.md)
- [ ] **H3** [Evaluation Methodology](books/04-ai-engineering-huyen/notes/ch03.md)
- [ ] **H4** [Evaluate AI Systems](books/04-ai-engineering-huyen/notes/ch04.md)
- [ ] **H5** [Prompt Engineering](books/04-ai-engineering-huyen/notes/ch05.md)
- [ ] **H6** [RAG and Agents](books/04-ai-engineering-huyen/notes/ch06.md)
- [ ] **H9** [Inference Optimization](books/04-ai-engineering-huyen/notes/ch09.md)
- [ ] **H10** [AI Engineering Architecture and User Feedback](books/04-ai-engineering-huyen/notes/ch10.md)

**Prove it →** [p4 — LLM eval harness](projects/p4-llm-eval-harness/)

**Exit criteria.** You have an eval suite you trust more than your own vibes, and you can
justify a model choice with numbers, latency, and cost — not preference.

---

## Phase 8 — Agents
**Weeks 27–30 · Lanham 1–11 · Core**

The least settled area in the field, so read it most critically. Frameworks will churn; the
decomposition — actions, memory, planning, reasoning, evaluation — will not.

- [ ] **L1** [Introduction to Agents and Their World](books/05-ai-agents-in-action-lanham/notes/ch01.md)
- [ ] **L2** [Harnessing the Power of Large Language Models](books/05-ai-agents-in-action-lanham/notes/ch02.md)
- [ ] **L3** [Engaging GPT Assistants](books/05-ai-agents-in-action-lanham/notes/ch03.md)
- [ ] **L4** [Exploring Multi-Agent Systems](books/05-ai-agents-in-action-lanham/notes/ch04.md)
- [ ] **L5** [Empowering Agents with Actions](books/05-ai-agents-in-action-lanham/notes/ch05.md)
- [ ] **L6** [Building Autonomous Assistants](books/05-ai-agents-in-action-lanham/notes/ch06.md)
- [ ] **L7** [Assembling and Using an Agent Platform](books/05-ai-agents-in-action-lanham/notes/ch07.md)
- [ ] **L8** [Understanding Agent Memory and Knowledge](books/05-ai-agents-in-action-lanham/notes/ch08.md)
- [ ] **L9** [Mastering Agent Prompts with Prompt Flow](books/05-ai-agents-in-action-lanham/notes/ch09.md)
- [ ] **L10** [Agent Reasoning and Evaluation](books/05-ai-agents-in-action-lanham/notes/ch10.md)
- [ ] **L11** [Agent Planning and Feedback](books/05-ai-agents-in-action-lanham/notes/ch11.md)

**Prove it →** [p5 — agent platform](projects/p5-agent-platform/) then
[p6 — production LLM app](projects/p6-production-llm-app/)

**Exit criteria.** You have an agent that recovers from a failed tool call instead of looping,
and you can say why it failed from your traces.

---

## If you are in a hurry

The 12-week compressed path, for someone who already writes production software:

| Weeks | Do this | Skip |
|---|---|---|
| 1–2 | G1–G4, G11 | G5–G9 (skim), G12–G14, G17–G19 |
| 3–5 | A1–A3, A6, A8 | A4, A5, A9 |
| 6–9 | **R1–R7 in full** | nothing — this is the core |
| 10–12 | H3, H4, H6, H9, H10 | H1, H2 (skim) |
| after | L5, L8, L10, L11 | the rest as needed |

Raschka is non-negotiable. Everything else can be revisited.

## If you are coming from a different angle

- **Already know classical ML?** Start at Phase 4. Read G16 first for the attention setup.
- **Already using LLM APIs at work?** Start at Phase 5 (to fix the black box) and Phase 7
  (to fix the evaluation), then backfill Phase 1 when a metric confuses you.
- **Want agents right now?** You will build brittle agents. Do Phase 5 and H3–H6 first;
  it is four weeks that saves you four months.

---

**Next:** [CURRICULUM.md](CURRICULUM.md) maps every concept across all five books, so you can
attack any single topic from intuition to implementation to production.
