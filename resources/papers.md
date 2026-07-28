# Papers

The primary sources behind the five books. Read the starred ones — they are short,
foundational, and the books are summarizing them.

---

## The architecture

- ⭐ **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017) — the
  transformer. Eight pages. Read it after
  [R3](../books/03-build-llm-from-scratch-raschka/notes/ch03.md), when the equations
  will read as familiar rather than cryptic.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (2015) — ResNet, the
  origin of the residual stream
- [GELU](https://arxiv.org/abs/1606.08415) (2016)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) (2021) — rotary positions
- [GQA](https://arxiv.org/abs/2305.13245) (2023) — grouped-query attention
- [FlashAttention](https://arxiv.org/abs/2205.14135) (2022) — same math, IO-aware

## Pretraining and scaling

- ⭐ **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**
  (2020) — GPT-3, and where in-context learning appeared
- [BERT](https://arxiv.org/abs/1810.04805) (2018)
- ⭐ **[Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)** (2022) —
  Chinchilla. The result that changed how models are sized.
- [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361) (2020)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) — an
  unusually detailed modern training report

## Post-training

- ⭐ **[InstructGPT](https://arxiv.org/abs/2203.02155)** (2022) — RLHF, and why
  models became usable
- ⭐ **[DPO](https://arxiv.org/abs/2305.18290)** (2023) — preference tuning without
  the RL loop
- [LIMA](https://arxiv.org/abs/2305.11206) (2023) — 1,000 examples can be enough
- [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022)

## Efficient adaptation

- ⭐ **[LoRA](https://arxiv.org/abs/2106.09685)** (2021)
- ⭐ **[QLoRA](https://arxiv.org/abs/2305.14314)** (2023) — 65B on one 48 GB GPU
- [GPTQ](https://arxiv.org/abs/2210.17323) (2022) · [AWQ](https://arxiv.org/abs/2306.00978) (2023)

## Retrieval

- ⭐ **[RAG](https://arxiv.org/abs/2005.11401)** (2020) — the original
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (2020)
- ⭐ **[Sentence-BERT](https://arxiv.org/abs/1908.10084)** (2019)
- [SimCSE](https://arxiv.org/abs/2104.08821) (2021) — positives from dropout alone
- ⭐ **[Lost in the Middle](https://arxiv.org/abs/2307.03172)** (2023) — why long
  context is not free
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (2024)

## Prompting and reasoning

- ⭐ **[Chain-of-Thought](https://arxiv.org/abs/2201.11903)** (2022)
- [Self-Consistency](https://arxiv.org/abs/2203.11171) (2022) — ensembling, renamed
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (2023)
- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) (2022)

## Agents

- ⭐ **[ReAct](https://arxiv.org/abs/2210.03629)** (2022) — reason + act
- [Toolformer](https://arxiv.org/abs/2302.04761) (2023)
- [Reflexion](https://arxiv.org/abs/2303.11366) (2023)
- ⭐ **[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)**
  (2024) — workflows vs agents, and why most things should be workflows
- [Generative Agents](https://arxiv.org/abs/2304.03442) (2023) — memory and reflection

## Evaluation

- ⭐ **[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)** (2023) — the
  paper that catalogued the biases
- [HELM](https://arxiv.org/abs/2211.09110) (2022)
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) (2024) — comparative evaluation
- [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004) (2023) — a
  useful corrective on discontinuous metrics

## Serving

- ⭐ **[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)** (2023)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022) — a rare free lunch
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (2022) —
  continuous batching

## Safety

- [Universal and Transferable Adversarial Attacks](https://arxiv.org/abs/2307.15043) (2023)
- [Prompt Injection](https://arxiv.org/abs/2302.12173) (2023) — indirect injection
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## Classical, still worth reading

- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) (2014)
- [Batch Normalization](https://arxiv.org/abs/1502.03167) (2015)
- [Adam](https://arxiv.org/abs/1412.6980) (2014) · [AdamW](https://arxiv.org/abs/1711.05101) (2017)
- [word2vec](https://arxiv.org/abs/1301.3781) (2013)

---

## How to read a paper

1. **Abstract, then figures, then conclusion.** Decide whether to continue.
2. **Skip the related work** on a first pass.
3. **Find the one idea.** Most papers have exactly one. Write it in a sentence.
4. **Check the baselines.** A large improvement over a weak baseline is not a
   large improvement.
5. **Read the limitations.** Where authors are honest, and where the next paper
   comes from.

Read the ⭐ papers *after* the corresponding chapter, not before. The book gives you
the scaffolding that makes the paper legible.
