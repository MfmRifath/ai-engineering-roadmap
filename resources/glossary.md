# Glossary

Terms that recur across the five books, defined once. Where two books use
different words for the same thing, both appear.

---

**Activation memory** — the cached forward-pass outputs backprop needs. Scales with
batch × sequence × depth, and is usually why training runs out of memory rather
than the weights. Gradient checkpointing trades compute for it.

**AdamW** — Adam with decoupled weight decay. The standard optimizer for
transformers, because Adam's L2 penalty interacts badly with its adaptive scaling.

**Attention** — a weighted average where the model learns the weights.
`softmax(QKᵀ/√d_k)V`. The only place information moves *between* positions in a
transformer.

**Base model** — pretrained only. Completes text; does not follow instructions.
Contrast **instruct model**.

**Bi-encoder** — encodes query and document separately, so document vectors can be
precomputed. Fast, less accurate. Contrast **cross-encoder**.

**BPE (byte pair encoding)** — iteratively merge the most frequent adjacent pair.
Starting from bytes makes out-of-vocabulary impossible.

**Causal mask** — sets attention to future positions to `-inf` *before* the softmax,
so a position can only attend to itself and earlier positions.

**Chunking** — splitting documents for retrieval. Size and overlap are
hyperparameters, and usually the highest-leverage ones in a RAG system.

**Compounding error** — per-step success rates multiply. 95% over ten steps is 60%.
The central constraint on multi-step LLM systems.

**Confused deputy** — an agent with legitimate credentials manipulated into using
them on an attacker's behalf. The core agent security risk.

**Context window** — the maximum sequence a model can attend over. Attention is
quadratic in it.

**Contextual embedding** — a token's representation *after* the transformer layers,
shaped by its neighbours. Contrast the static embedding matrix.

**Contrastive learning** — train so that positive pairs are close and negatives far.
The recipe behind word2vec, CLIP, and sentence embeddings alike.

**Cross-encoder** — processes query and document *together*, with attention across
both. Much more accurate, far too slow for a whole corpus. Used as a reranker.

**Cross-entropy** — the loss for next-token prediction. Multiclass classification
over the vocabulary. `exp(cross_entropy)` is perplexity.

**Data snooping** — letting the test set influence your decisions. Your brain
overfits even when your code does not.

**Decode** — generating tokens one at a time. **Memory-bandwidth-bound**, because
every weight must be read per token. Contrast **prefill**.

**Distillation** — training a small model on a large model's outputs.

**DPO** — Direct Preference Optimization. Preference tuning without a reward model
or an RL loop.

**Efficiency failure** — the right answer via far more steps and cost than
necessary. Invisible to end-to-end accuracy; visible in trajectory evaluation.

**Embedding** — a dense learned vector. An `nn.Embedding` layer is a lookup table of
shape (vocab, dim).

**Emergence** — capabilities appearing at scale without being trained for. Describes
an observation; does not explain one, and some cases are metric artifacts.

**FlashAttention** — an IO-aware tiled attention implementation. Identical
mathematics, much less memory traffic.

**Functional correctness** — does the generated code pass the tests, does the SQL
return the right rows. The gold standard of evaluation where it applies.

**Goodput** — requests per second that meet your latency SLO. More honest than raw
throughput.

**GQA** — grouped-query attention. Query head groups share key/value heads, shrinking
the KV cache. The current standard.

**Guardrails** — input and output filters. Input catches injection and abuse before
you pay for inference; output catches unsafe or malformed generations.

**Hallucination** — the model samples a plausible continuation. It has no truth
predicate, and post-training rewarded answering over abstaining.

**Hard negative** — a plausible-but-wrong example. Where a contrastive model actually
learns the boundary; random negatives stop producing gradient quickly.

**Indirect prompt injection** — malicious instructions inside content your system
*retrieves* rather than content the user types. The dangerous variant for agents.

**Inductive bias** — assumptions an architecture encodes. CNNs assume locality;
transformers assume almost nothing, which is why they need more data and then win.

**Instruct model** — post-trained (SFT + preference tuning) to follow instructions.

**KV cache** — cached keys and values for previous positions, so each new token does
not re-run the prefix. At scale it, not the weights, limits concurrency.

**Layer normalization** — normalizes across features within one example. Independent
of batch size, which is why transformers use it rather than BatchNorm.

**Least privilege** — an agent cannot misuse a tool it does not have. The strongest
injection defense, because it is structural rather than probabilistic.

**LLM-as-a-judge** — using a model to evaluate model output. Practical, and biased by
position, verbosity, and self-preference. Must be validated against humans.

**LoRA** — freeze W, learn a low-rank update BA. ~0.1–1% trainable parameters, so the
optimizer state largely disappears.

**Lost in the middle** — models attend less reliably to the middle of a long context.
Put important content at the edges.

**Manifold hypothesis** — real high-dimensional data concentrates on a much
lower-dimensional surface. Why embeddings work at all.

**MMR** — maximal marginal relevance. Select results that are relevant *and* mutually
diverse.

**Perplexity** — `exp(cross_entropy)`. The effective number of tokens being chosen
among. Only comparable across models sharing a tokenizer.

**PEFT** — parameter-efficient fine-tuning. LoRA and friends.

**Prefill** — processing the whole prompt in parallel. **Compute-bound**, and it
determines time-to-first-token.

**Pre-norm** — normalize before the sublayer, then add to the residual. Trains far
more stably at depth than the original post-norm.

**Prompt injection** — text that gets treated as instructions. Unsolvable at the
model layer, because everything in the context is one token sequence.

**QLoRA** — LoRA over a 4-bit quantized frozen base. Puts a 7B fine-tune on 16 GB.

**Quantization** — storing weights in fewer bits with a scale per group. int8 nearly
free, int4 a small cost, below that it degrades.

**RAG** — retrieval-augmented generation. Retrieve, put in the prompt, generate.
Retrieval quality is the ceiling on system quality.

**Recall@k** — of the relevant documents, how many appeared in the top k. The metric
that matters most for RAG.

**Reflection** — periodically summarizing a session into durable memories, rather
than storing raw turns.

**Residual stream** — the `x` carried through `x = x + sublayer(norm(x))`. A shared
bus each block reads from and writes to.

**RLHF** — reinforcement learning from human feedback. Train a reward model on
preferences, optimize the policy against it with a KL penalty.

**RoPE** — rotary positional embeddings. Rotate Q and K by angle proportional to
position, so attention depends on *relative* distance.

**RRF** — reciprocal rank fusion. Combine rankings using only ranks, so incomparable
score scales never matter.

**Sampling** — choosing a token from the output distribution. Temperature, top-k,
top-p. The source of LLM non-determinism.

**Self-supervised** — labels manufactured from the input. Next-token prediction is
the canonical case, and the economic basis of the field.

**SetFit** — competitive classification from ~8 examples per class, by turning
classification into a similarity problem to multiply the labels.

**SFT** — supervised fine-tuning on (instruction, response) pairs.

**Speculative decoding** — a draft model proposes tokens, the large model verifies
them in one pass. Output is mathematically identical.

**Superposition** — more features than neurons, encoded in overlapping directions.
What sparse autoencoders try to disentangle.

**Temperature** — divides logits before the softmax. Adds randomness, never
competence.

**Trajectory** — the full sequence of an agent's steps. The unit of agent
evaluation, because the answer alone hides too much.

**Training/serving skew** — preprocessing drifts between training and production.
Nothing crashes; quality quietly degrades.

**Weight tying** — the output head shares the token embedding matrix. Saves 38.6M
parameters in GPT-2 small, and says the two spaces share a geometry.
