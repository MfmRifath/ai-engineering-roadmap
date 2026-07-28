# Numbers

Arithmetic you should be able to do in your head, or in
[`aieng.serving.budget`](../src/aieng/serving/budget.py).

---

## Model memory

**Inference ≈ parameters × bytes per parameter**

| Precision | Bytes | 7B | 70B |
|---|---|---|---|
| fp32 | 4 | 28 GB | 280 GB |
| fp16 / bf16 | 2 | **14 GB** | 140 GB |
| int8 | 1 | 7 GB | 70 GB |
| int4 | 0.5 | **3.5 GB** | 35 GB |

Add the KV cache. It is often larger than you expect.

---

## Training memory — where it actually goes

For 7B in fp16 with AdamW:

| Component | Formula | Size |
|---|---|---|
| Weights | 2 B/param | 14 GB |
| Gradients | 2 B/trainable | 14 GB |
| **Adam states** | **8 B/trainable** (two fp32 moments) | **56 GB** |
| Activations | batch × seq × layers | varies |
| **Total** | | **~90 GB+** |

**Adam's optimizer state is the villain** — four times the weight memory. That is
why PEFT works: eliminate trainable parameters and it vanishes with them.

| Method | 7B total |
|---|---|
| Full fine-tune | ~90 GB |
| LoRA (0.5% trainable) | ~14 GB |
| **QLoRA (nf4 base + LoRA)** | **~4 GB** ← fits a 16 GB card |

→ [H7](../books/04-ai-engineering-huyen/notes/ch07.md)

---

## KV cache

```
2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes
```

The 2 is keys **and** values. Note `n_kv_heads`, not `n_heads` — that difference
is the entire point of GQA.

Llama-3-8B (32 layers, 8 KV heads, head_dim 128), fp16:

| Context | batch 1 | batch 32 |
|---|---|---|
| 4k | 0.5 GB | 17 GB |
| 8k | **1.1 GB** | **34 GB** ← more than the 16 GB of weights |
| 32k | 4.3 GB | 137 GB |

**At scale the cache, not the model, limits your concurrency.**

---

## Decode speed floor

Generating one token requires reading **every weight** from memory. So:

```
ms per token ≥ (params × bytes) / memory_bandwidth
```

At 2 TB/s:

| Model | Precision | Floor | Ceiling |
|---|---|---|---|
| 7B | fp16 | 7.0 ms | ~143 tok/s |
| 7B | int4 | 1.75 ms | ~571 tok/s |
| 70B | fp16 | 70 ms | ~14 tok/s |

If you measure far above the floor, the problem is your serving stack, not your
hardware. **Quantization speeds up decode because it halves memory traffic**, not
because the arithmetic got cheaper.

→ [H9](../books/04-ai-engineering-huyen/notes/ch09.md)

---

## Tokens

| Rule of thumb | Value |
|---|---|
| English words per token | ~0.75 |
| Characters per token (English) | ~4 |
| A page of prose | ~500 tokens |
| Non-English penalty | **3–5× more tokens** for the same meaning |

That last row is a real production cost difference for multilingual apps.
→ [A2](../books/02-hands-on-llms-alammar/notes/ch02.md)

---

## Agent cost

**Not** `steps × single_call_cost`. Context grows every step, so input tokens
accumulate roughly quadratically:

```
total_input = Σ(base + per_step × i) for i in 0..steps-1
```

10 steps, 2k system prompt, 500 tokens/step, at $3/$15 per M ≈ **$0.20 per task.**
At 1,000 tasks/day that is $6,000/month. Model it before deploying.

→ [L2](../books/05-ai-agents-in-action-lanham/notes/ch02.md)

---

## Compounding reliability

Success rates **multiply**:

| Per-step | 4 steps | 10 steps | 20 steps |
|---|---|---|---|
| 90% | 66% | 35% | 12% |
| 95% | 81% | **60%** | 36% |
| 99% | 96% | 90% | 82% |

This is why production "agents" are mostly constrained workflows with one or two
model decision points. The constraint is a feature.

→ [H6](../books/04-ai-engineering-huyen/notes/ch06.md)

---

## GPT-2 small, for reference

| | |
|---|---|
| Parameters | 124M (163M untied — weight tying saves 38.6M) |
| Layers / heads / d_model | 12 / 12 / 768 |
| head_dim | 64 |
| Context | 1024 |
| Vocabulary | 50,257 |
| Embedding matrix | 50257 × 768 = 38.6M (~31% of the model) |
| FFN expansion | 4× → 3072 |
| Parameters in FFNs | ~2/3 of non-embedding |

→ [R4](../books/03-build-llm-from-scratch-raschka/notes/ch04.md)

---

## Perplexity

`perplexity = exp(cross_entropy)` — the effective number of tokens being chosen
among.

| Loss | Perplexity | Means |
|---|---|---|
| 10.8 | ~50,000 | Untrained (uniform over the vocabulary) |
| 4.0 | ~55 | Learning |
| 3.0 | ~20 | Reasonable small model |
| 2.0 | ~7.4 | Good |

Only comparable across models with the **same tokenizer**. Use bits-per-byte
otherwise.
→ [H3](../books/04-ai-engineering-huyen/notes/ch03.md)
