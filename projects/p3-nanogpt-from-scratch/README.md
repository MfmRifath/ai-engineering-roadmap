# p3 — nanoGPT from scratch

**After Phase 5 · Draws on [R2](../../books/03-build-llm-from-scratch-raschka/notes/ch02.md)–[R5](../../books/03-build-llm-from-scratch-raschka/notes/ch05.md)**

Assemble chapters 2–5 into one repository that trains a GPT from zero on a corpus
you chose. **No `transformers` modeling code** — you may use `tiktoken` for
tokenization if you prefer, but every layer is yours.

## Why this project

Raschka's chapters build the pieces. This makes you own the whole thing: data
pipeline, model, training loop, checkpointing, generation, and evaluation. That
is the difference between having followed a tutorial and being able to build one.

## Definition of done

- [ ] Trains from a plain text file to a generating model with one command
- [ ] Uses **your** attention, block, and GPT from
      [`aieng.transformer`](../../src/aieng/transformer/) — and
      `pytest tests/test_attention.py` passes
- [ ] Training loop with AdamW, **gradient clipping, LR warmup and cosine decay**
      ([`aieng.nn.schedules`](../../src/aieng/nn/schedules.py))
- [ ] Logs train and validation loss, and **perplexity**, on a schedule
- [ ] Checkpoints model **and optimizer** state, and can resume
- [ ] Generates with temperature, top-k, and top-p
- [ ] **Loads OpenAI's GPT-2 weights into your architecture** and generates
      coherent text — the proof your implementation is correct
- [ ] A README with your loss curve and a sample of generated text

## The moment that matters

Loading OpenAI's released GPT-2 weights into *your* GPT class and watching it
produce fluent English. Nothing else in the roadmap is as convincing.

It is fiddly on purpose: OpenAI's checkpoint uses TensorFlow naming, packs Q/K/V
into a single fused matrix, and uses transposed weight conventions. **Assert every
shape on assignment** — a silent mismatch gives you a model that runs and emits
nonsense, which is a miserable bug to chase.

## Pitfalls

- **Off-by-one in the targets.** `targets = tokens[i+1 : i+max_length+1]`. Get it
  wrong and the model learns to copy its input — suspiciously low loss.
- **Forgetting `optimizer.zero_grad()`.** PyTorch accumulates gradients.
- **Passing probabilities to `cross_entropy`.** It wants raw logits.
- **Biased variance in LayerNorm.** Use `unbiased=False` or loaded GPT-2 weights
  drift.
- **Post-norm instead of pre-norm.** Trains much worse at depth.
- **Expecting good text from a small corpus.** A 124M model on 20k tokens will
  overfit and memorize. That is the demonstration, not a failure.

## Suggested corpora

Anything you can read yourself, so you can judge the output: your own writing,
a public-domain book from Project Gutenberg, your commit messages, a chat export.
Small and personal beats large and generic for this exercise — you will recognize
what it learned.

## Stretch

- **RoPE** instead of learned absolute positions, then generate past the training
  context length and see what happens.
- **KV caching** in generation. Measure tokens/sec with and without, and plot the
  ratio against sequence length.
- Compare your measured tokens/sec against
  [`decode_floor_ms`](../../src/aieng/serving/budget.py). **Explain the gap.**
- **LoRA** from [`aieng.finetune.lora`](../../src/aieng/finetune/lora.py), then
  fine-tune on a second corpus and compare trainable parameter counts.
- Flash-attention via `F.scaled_dot_product_attention` — confirm identical output,
  measure the speedup.
- Scale to GPT-2 medium by changing only the config.

## Getting started

```bash
pip install -e ".[llm]"
python -m projects.p3_nanogpt.train --data data/corpus.txt --config gpt2-small
python -m projects.p3_nanogpt.generate --checkpoint out/ckpt.pt --prompt "Once upon a" --temperature 0.8 --top-p 0.9
python -m projects.p3_nanogpt.load_gpt2 --size 124M   # the proof
```
