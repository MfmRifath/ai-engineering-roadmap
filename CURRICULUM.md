# The Concept Matrix

The reason to study five books instead of one: almost every important idea appears in three or
four of them, each time from a different angle. Alammar shows you the picture, Raschka makes you
build it, Huyen tells you what breaks in production, Géron gives you the statistical footing.

This page maps every major concept to every book that covers it, so you can take **one topic**
and spiral through it — intuition → implementation → operation — instead of reading five books
end to end and hoping they connect.

**Keys:** `G` Géron · `A` Alammar · `R` Raschka · `H` Huyen · `L` Lanham
**Depth:** ●●● implement it · ●●○ working understanding · ●○○ mentioned in passing

---

## Foundations

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Train/val/test discipline | **G2** ●●● | — | R6 ●●○ | H4 ●●○ | — | G2 |
| Overfitting & regularization | **G4, G11** ●●● | — | R5 ●●○ | H7 ●○○ | — | G4 |
| Gradient descent & optimizers | **G4, G11** ●●● | — | **R5** ●●● | — | — | G4 |
| Backpropagation | **G10–G12** ●●● | — | R App. A ●●● | H7 ●●○ | — | G10 |
| Loss functions | **G4, G10** ●●● | A11 ●●○ | **R5** ●●● | H3 ●●○ | — | G4 |
| Bias / variance | **G4** ●●● | — | — | H4 ●○○ | — | G4 |
| Feature engineering | **G2** ●●● | — | — | H8 ●●○ | — | G2 |

## Representation

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Tokenization (BPE) | G16 ●○○ | **A2** ●●● | **R2** ●●● | H2 ●●○ | — | A2 → R2 |
| Token embeddings | G16 ●●○ | **A2** ●●● | **R2** ●●● | H3 ●●○ | — | A2 |
| Text/sentence embeddings | — | **A2, A10** ●●● | — | **H3** ●●● | L8 ●●○ | A10 |
| word2vec & contrastive training | G17 ●○○ | **A2, A10** ●●● | — | — | — | A2 |
| Dimensionality reduction | **G8** ●●● | A5 ●●○ | — | — | — | G8 |
| Autoencoders / latent space | **G17** ●●● | A5 ●○○ | — | — | — | G17 |

## The Transformer

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Why RNNs fail on long context | **G15, G16** ●●● | A1 ●●○ | R1 ●●○ | — | — | G16 |
| Attention (the mechanism) | G16 ●●○ | **A3** ●●● | **R3** ●●● | H2 ●●○ | — | A3 → R3 |
| Self- vs cross-attention | G16 ●●○ | A3 ●●○ | **R3** ●●● | — | — | R3 |
| Causal masking | — | A3 ●●○ | **R3** ●●● | — | — | R3 |
| Multi-head attention | G16 ●○○ | A3 ●●○ | **R3** ●●● | — | — | R3 |
| Transformer block (full) | G16 ●●○ | **A3** ●●● | **R4** ●●● | H2 ●●○ | — | A3 → R4 |
| Layer norm, residuals, GELU | — | A3 ●●○ | **R4** ●●● | — | — | R4 |
| Positional encoding / RoPE | G16 ●○○ | **A3** ●●● | **R2, R4** ●●● | — | — | A3 |
| Encoder- vs decoder-only | — | **A1, A4** ●●● | R1 ●●○ | H2 ●●○ | — | A1 |
| KV caching | — | **A3** ●●● | — | **H9** ●●● | — | A3 → H9 |
| Efficient attention (Flash, GQA) | — | **A3** ●●● | — | **H9** ●●● | — | H9 |
| Scaling laws & model size | — | A1 ●○○ | R1 ●●○ | **H2** ●●● | — | H2 |

## Generation

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Sampling: greedy, top-k, top-p | G16 ●●○ | **A6** ●●● | **R5** ●●● | **H2** ●●● | L2 ●●○ | R5 → H2 |
| Temperature | G17 ●○○ | A6 ●●○ | **R5** ●●● | **H2** ●●● | L2 ●●○ | R5 |
| Structured / constrained output | — | **A6** ●●● | — | **H2** ●●● | L5 ●●● | H2 |
| Test-time compute | — | A6 ●●○ | — | **H2** ●●● | L10 ●●○ | H2 |
| Hallucination (why it happens) | — | A6 ●●○ | — | **H2, H3** ●●● | L10 ●●○ | H2 |

## Training & Adaptation

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Pretraining objective | — | A1 ●●○ | **R5** ●●● | **H2** ●●● | — | R5 |
| Supervised fine-tuning (SFT) | G11 ●●○ | **A12** ●●● | **R6, R7** ●●● | **H7** ●●● | — | R7 |
| Instruction tuning | — | A12 ●●○ | **R7** ●●● | H2 ●●○ | L2 ●○○ | R7 |
| Preference tuning / RLHF / DPO | G18 ●○○ | **A12** ●●● | — | **H2, H7** ●●● | — | H2 |
| Reinforcement learning basis | **G18** ●●● | — | — | H2 ●○○ | L11 ●●○ | G18 |
| PEFT / LoRA / QLoRA | — | **A11, A12** ●●● | R App. E ●●● | **H7** ●●● | — | H7 → R App. E |
| Quantization & precision | — | A7, A12 ●●○ | — | **H7, H9** ●●● | L2 ●○○ | H7 |
| Memory math for training | — | A12 ●●○ | R5 ●●○ | **H7** ●●● | — | H7 |
| Model merging / distillation | — | — | — | **H7, H8** ●●● | — | H7 |
| Transfer learning | **G11, G14** ●●● | A11 ●●● | R6 ●●● | H7 ●●○ | — | G11 |

## Data

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Data pipelines & loaders | **G13** ●●● | — | **R2** ●●● | H8 ●●○ | — | G13 |
| Data quality & coverage | G2 ●●○ | — | — | **H8** ●●● | — | H8 |
| Synthetic data generation | — | A12 ●●○ | R7 ●●○ | **H8** ●●● | — | H8 |
| Deduplication & filtering | — | — | R5 ●○○ | **H8** ●●● | — | H8 |
| Annotation & labeling | G3 ●●○ | A4 ●●○ | — | **H8** ●●● | — | H8 |

## Retrieval

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Semantic / dense search | — | **A8** ●●● | — | **H6** ●●● | L8 ●●○ | A8 |
| Sparse retrieval & BM25 | — | A8 ●●○ | — | **H6** ●●● | — | H6 |
| Hybrid retrieval & reranking | — | **A8** ●●● | — | **H6** ●●● | — | A8 → H6 |
| Chunking strategy | — | A8 ●●○ | — | **H6** ●●● | L8 ●●○ | H6 |
| RAG architecture end to end | — | **A8** ●●● | — | **H6, H10** ●●● | L8 ●●○ | A8 → H6 |
| Vector indexes | — | A8 ●●○ | — | **H6** ●●● | L8 ●●○ | H6 |
| RAG vs fine-tuning (choosing) | — | — | — | **H7** ●●● | — | H7 |

## Evaluation

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Classification metrics | **G3** ●●● | A4 ●●● | R6 ●●● | H4 ●●○ | — | G3 |
| Cross-validation | **G2, G3** ●●● | — | — | — | — | G2 |
| Perplexity & entropy | — | A1 ●○○ | **R5** ●●● | **H3** ●●● | — | H3 |
| Functional correctness | — | — | — | **H3** ●●● | L10 ●●○ | H3 |
| Similarity to reference | — | A10 ●●○ | — | **H3** ●●● | — | H3 |
| LLM-as-a-judge | — | — | — | **H3** ●●● | **L10** ●●● | H3 |
| Comparative / pairwise ranking | — | — | — | **H3** ●●● | — | H3 |
| Building an eval pipeline | G2 ●●○ | — | — | **H4** ●●● | L10 ●●○ | H4 |
| Model selection & benchmarks | **G2** ●●● | A4 ●●○ | — | **H4** ●●● | L2 ●●○ | H4 |
| Agent-specific evaluation | — | — | — | H6 ●●○ | **L10** ●●● | L10 |

## Prompting

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| In-context / few-shot learning | — | **A6** ●●● | R7 ●●○ | **H5** ●●● | L2 ●●○ | A6 |
| System vs user prompts | — | A6 ●●○ | — | **H5** ●●● | **L2** ●●● | H5 |
| Chain-of-thought | — | **A6** ●●● | — | H5 ●●○ | **L10** ●●● | A6 |
| Self-consistency & tree-of-thought | — | **A6** ●●● | — | H5 ●○○ | L10 ●●● | A6 |
| Prompt chaining | — | **A7** ●●● | — | H5 ●●○ | **L9** ●●● | A7 |
| Prompt versioning & management | — | — | — | **H5** ●●● | **L9** ●●● | H5 |
| Prompt injection & jailbreaks | — | — | — | **H5** ●●● | L5 ●●○ | H5 |
| Defensive prompting / guardrails | — | A6 ●○○ | — | **H5, H10** ●●● | L5 ●●○ | H5 |

## Agents

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| What an agent *is* | — | A7 ●●○ | — | **H6** ●●● | **L1** ●●● | H6 → L1 |
| Tool use / function calling | — | A7 ●●○ | — | **H6** ●●● | **L5** ●●● | L5 |
| Agent memory | — | A7 ●●○ | — | H6 ●●○ | **L8** ●●● | L8 |
| Planning | — | — | — | **H6** ●●● | **L11** ●●● | H6 → L11 |
| Reasoning loops (ReAct etc.) | — | A6 ●●○ | — | H6 ●●○ | **L10, L11** ●●● | L10 |
| Multi-agent systems | — | — | — | H6 ●○○ | **L4, L7** ●●● | L4 |
| Failure modes & recovery | — | — | — | **H6** ●●● | **L10** ●●● | H6 |
| Feedback loops | — | — | — | **H10** ●●● | **L11** ●●● | L11 |

## Shipping

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| Serving a model | **G19** ●●● | — | — | **H9** ●●● | L7 ●●○ | G19 → H9 |
| Latency & throughput metrics | G19 ●●○ | — | — | **H9** ●●● | — | H9 |
| Batching & continuous batching | G19 ●○○ | — | — | **H9** ●●● | — | H9 |
| Caching | — | A3 ●●○ | — | **H9, H10** ●●● | — | H10 |
| Routing & gateways | — | — | — | **H10** ●●● | L7 ●●○ | H10 |
| Monitoring & observability | G19 ●●○ | — | — | **H10** ●●● | L10 ●●○ | H10 |
| User feedback design | — | — | — | **H10** ●●● | L11 ●●● | H10 |
| Cost management | — | A1 ●○○ | — | **H4, H9** ●●● | L2 ●●○ | H4 |
| GPU / accelerator basics | G19 ●●○ | A1 ●○○ | R5 ●●○ | **H9** ●●● | — | H9 |

## Multimodal

| Concept | Géron | Alammar | Raschka | Huyen | Lanham | Start here |
|---|---|---|---|---|---|---|
| CNNs & vision | **G14** ●●● | A9 ●●○ | — | — | — | G14 |
| Vision-language models / CLIP | — | **A9** ●●● | — | H2 ●○○ | L6 ●○○ | A9 |
| Generative image models | **G17** ●●● | A9 ●●○ | — | H1 ●○○ | — | A9 |
| Multimodal RAG | — | A9 ●●○ | — | **H6** ●●● | — | H6 |

---

## Study spirals

Pick a concept, walk the spiral. Each pass adds a layer the previous one could not.

### Spiral: Attention
1. **G16** — feel the problem. Watch an RNN lose the beginning of a long sentence.
2. **A3** — see the solution. Alammar's diagrams; do not move on until Q, K, V feel obvious.
3. **R3** — build it. ~200 lines, four escalating versions ending in causal multi-head.
4. **H9** — operate it. KV cache, GQA, FlashAttention — why the naive version is too slow.
5. **Verify:** `pytest tests/test_attention.py` — yours must match PyTorch's within 1e-5.

### Spiral: RAG
1. **A8** — build the naive version. Embed, index, retrieve, stuff into a prompt. It works!
2. **H6** — learn why it does not work. Chunk boundaries, hybrid retrieval, reranking, the
   retrieval quality ceiling nobody measures.
3. **H3–H4** — measure it. Retrieval and generation are two separate eval problems.
4. **L8** — extend it. Retrieval as agent memory rather than a one-shot lookup.
5. **Ship:** [p2 — RAG over my library](projects/p2-rag-over-my-library/)

### Spiral: Evaluation
1. **G2–G3** — the statistical basis: cross-validation, precision/recall, why a single number lies.
2. **H3** — the LLM-specific problem: open-ended output has no ground truth. Perplexity,
   functional correctness, similarity, AI-as-judge and its failure modes.
3. **H4** — the systems problem: evaluate *components*, write a guideline, pick a model.
4. **L10** — the agent problem: evaluating a trajectory, not an output.
5. **Ship:** [p4 — LLM eval harness](projects/p4-llm-eval-harness/)

### Spiral: Fine-tuning
1. **G11** — transfer learning, the original idea: freeze the bottom, retrain the top.
2. **R6** then **R7** — do it by hand: classification head, then instruction following.
3. **A11–A12** — the modern toolchain: PEFT, LoRA, quantized bases.
4. **H7** — the decision itself: when *not* to. Then **H8**, because the dataset is the work.
5. **Verify:** you can compute the VRAM for a 7B LoRA run before launching it.

---

## How to use this page

- **Stuck on a concept?** Find the row. Read the ●●● cell you have not read yet.
- **Reviewing?** Cover the book columns and try to name where each concept lives.
- **Planning a week?** Pick a spiral, not a chapter. Spirals produce understanding;
  chapters produce notes.

Back to [ROADMAP.md](ROADMAP.md) · [README](README.md)
