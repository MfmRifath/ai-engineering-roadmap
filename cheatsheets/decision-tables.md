# Decision tables

The choices you will make repeatedly. Each row links to the chapter that argues it.

---

## RAG or fine-tuning?

| The model... | Do this |
|---|---|
| doesn't **know** something | **RAG** — updatable, citable, auditable |
| doesn't **behave** how you want | **Fine-tune** — format, style, domain patterns |
| both | RAG first, then fine-tune the behaviour |
| is too slow or expensive | Fine-tune a **smaller** model to replace a prompted large one |

**Default is neither.** Try prompting properly first.
→ [H7](../books/04-ai-engineering-huyen/notes/ch07.md)

---

## How should I classify text?

| Situation | Approach | Cost |
|---|---|---|
| A fine-tuned model for your exact task exists | Use it | Tiny |
| 100s–1000s of labels | **Frozen embeddings + logistic regression** ← default | Small |
| 10–100 labels | **SetFit** | Small |
| No labels | Zero-shot via embedded label descriptions | Small |
| Labels change weekly, or need reasoning | Prompt a generative model | Large |
| No labels, lots of domain text | Continued MLM pretraining, then any of the above | Medium |

Start at the top. Climb only when the rung below measurably fails.
→ [A4](../books/02-hands-on-llms-alammar/notes/ch04.md),
[A11](../books/02-hands-on-llms-alammar/notes/ch11.md)

---

## Which classification metric?

| You care most about | Optimize | Example |
|---|---|---|
| Not crying wolf | **Precision** | Content filter, auto-delete |
| Not missing anything | **Recall** | Fraud, disease screening, safety |
| Genuinely neither | F1 | Rare — usually you are avoiding a decision |
| Ranking quality, rare positives | **PR AUC** | Not ROC AUC |
| Ranking quality, balanced | ROC AUC | |

**Never accuracy on imbalanced data.**
→ [G3](../books/01-hands-on-ml-geron/notes/ch03.md)

---

## Which retrieval method?

| Query type | Winner |
|---|---|
| Exact IDs, codes, rare names | **BM25** |
| Paraphrase, meaning | **Dense** |
| Real traffic | **Hybrid + RRF** ← default |
| Need the best top-5 | Hybrid, then a **cross-encoder reranker** |

→ [A8](../books/02-hands-on-llms-alammar/notes/ch08.md),
[H6](../books/04-ai-engineering-huyen/notes/ch06.md)

---

## Which evaluation method?

| Output shape | Method | Fails when |
|---|---|---|
| Verifiable (code, SQL, JSON) | **Functional correctness** ← restructure tasks to reach this | Rarely |
| Has a gold reference | Similarity | Good answers are phrased differently |
| Open-ended | LLM judge | Judge is biased, inconsistent, self-preferring |
| Comparing two systems | **Pairwise comparison** | You need an absolute bar |
| A ranked list | recall@k, MRR, nDCG | |
| An agent | **Trajectory evaluation** | You only look at the answer |

→ [H3](../books/04-ai-engineering-huyen/notes/ch03.md),
[L10](../books/05-ai-agents-in-action-lanham/notes/ch10.md)

---

## Sampling parameters

| Task | temperature | top-p |
|---|---|---|
| Extraction, classification, tool calls | **0** | 1.0 |
| Factual Q&A | 0–0.3 | 0.9 |
| General writing | 0.7 | 0.9 |
| Creative | 1.0+ | 0.95 |
| Self-consistency sampling | 0.7 | 0.9 |

`temperature=0` is not bit-reproducible. Raising temperature adds randomness,
never competence.
→ [A6](../books/02-hands-on-llms-alammar/notes/ch06.md)

---

## Buy or self-host?

| | API | Self-hosted |
|---|---|---|
| Low volume | **Cheaper** | Fixed cost dominates |
| High volume | Expensive | **Cheaper** |
| Data must not leave | No | **Yes** |
| Version stability | **Changes under you** | Pinned |
| Ops burden | None | Real and permanent |

Compute your crossover volume rather than arguing about it.
→ [H4](../books/04-ai-engineering-huyen/notes/ch04.md)

---

## How much autonomy?

| Level | Pattern | Use when |
|---|---|---|
| 1 | Fixed chain | The path is knowable ← **most production value** |
| 2 | Routing | The model picks one of N fixed paths |
| 3 | Tool loop | Genuinely dynamic decisions |
| 4 | Planning + execution | Multi-step, plan is reviewable |
| 5 | Fully autonomous | Almost never |

Ask: **what is the lowest level that solves this?**
→ [L1](../books/05-ai-agents-in-action-lanham/notes/ch01.md)

---

## Fixing a model that underperforms

| Symptom | First thing to try |
|---|---|
| Underfits (train and val both bad) | Bigger model, better features, less regularization |
| Overfits (big train/val gap) | More data, regularization, simpler model, early stopping |
| Good offline, bad in production | **Data mismatch** — fix the data, not the model |
| RAG gives wrong answers | Measure **retrieval** first — it is usually there |
| Agent gets right answers slowly | Efficiency failure — reference trajectories |
| Inconsistent output | You are sampling. Set temperature 0. |

→ [G1](../books/01-hands-on-ml-geron/notes/ch01.md),
[G4](../books/01-hands-on-ml-geron/notes/ch04.md)
