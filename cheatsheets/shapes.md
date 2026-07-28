# Tensor shapes

Where most transformer bugs live. `B` = batch, `T` = sequence length,
`C` = `d_model`, `H` = heads, `Dh` = `C / H` = head dimension, `V` = vocabulary.

---

## The full forward pass

```
input_ids                                    (B, T)          int64
  │
  ├─ tok_emb(input_ids)                      (B, T, C)
  ├─ pos_emb(arange(T))                      (T, C)   ──┐ broadcast
  └─ x = tok + pos                           (B, T, C) ←─┘
  │
  ├─ TransformerBlock × n_layers             (B, T, C)   ← shape-preserving,
  │                                                        which is what lets
  │                                                        you stack them
  ├─ final_norm(x)                           (B, T, C)
  └─ out_head(x)                             (B, T, V)   ← logits

generation uses logits[:, -1, :]             (B, V)      ← ONLY the last position
```

---

## Inside attention

```
x                                            (B, T, C)

W_query(x)                                   (B, T, C)
  .view(B, T, H, Dh)                         (B, T, H, Dh)   ← split heads
  .transpose(1, 2)                           (B, H, T, Dh)   ← heads to the front

q @ k.transpose(-2, -1)                      (B, H, T, T)    ← the score matrix
                                                               (this is the n²)
  masked_fill(mask[:T,:T], -inf)             (B, H, T, T)
  softmax(scores / Dh**0.5, dim=-1)          (B, H, T, T)    ← rows sum to 1

attn @ v                                     (B, H, T, Dh)
  .transpose(1, 2)                           (B, T, H, Dh)
  .contiguous().view(B, T, C)                (B, T, C)       ← merge heads
  out_proj(...)                              (B, T, C)
```

**Three things that bite:**

1. `.view()` fails after `.transpose()` — the tensor is non-contiguous. Use
   `.contiguous().view()` or `.reshape()`.
2. Scale by `Dh**0.5`, **not** `C**0.5`. The dot products are per head.
3. Slice the mask to `[:T, :T]` — the buffer is `context_length` square and your
   sequence is usually shorter.

---

## Loss

```
logits                                       (B, T, V)
targets                                      (B, T)

cross_entropy(
    logits.flatten(0, 1),                    (B*T, V)
    targets.flatten(),                       (B*T,)
)                                            scalar
```

`cross_entropy` wants **raw logits**. It applies log-softmax internally, stably.
Passing softmax output trains badly and raises nothing.

---

## The data pipeline

```
token_ids                                    (N,)     the whole corpus

sliding window, stride s:
  inputs[i]  = tokens[i     : i+T]           (T,)
  targets[i] = tokens[i + 1 : i+T+1]         (T,)     ← shifted by ONE

batched                                      (B, T)
```

That off-by-one is the single most common bug in chapter 2. Get it wrong and the
model learns to copy its input, which shows up as suspiciously low loss.

---

## KV cache

```
per layer, cached:
  keys    (B, H_kv, T_so_far, Dh)
  values  (B, H_kv, T_so_far, Dh)

each new token contributes:
  (B, H_kv, 1, Dh)  →  appended
```

`H_kv` may be smaller than `H` — that is grouped-query attention, and shrinking
this tensor is its entire purpose.

Memory = `2 × n_layers × H_kv × Dh × T × B × bytes`.

---

## Parameter counts

Per transformer block, with `C = d_model`:

| Component | Parameters |
|---|---|
| Attention (Q, K, V, out) | `4 × C²` |
| FFN (up, down, 4× expansion) | `8 × C²` |
| LayerNorms | `4 × C` (negligible) |
| **Per block** | **`~12 × C²`** |

Plus embeddings: `V × C` (tied) or `2 × V × C` (untied).

GPT-2 small: `12 blocks × 12 × 768² ≈ 85M`, plus `50257 × 768 = 38.6M` embeddings
≈ 124M. **The FFN is two-thirds of the block.**

---

## Common shape errors, decoded

| Error | Meaning |
|---|---|
| `view size is not compatible with input tensor's size and stride` | Missing `.contiguous()` after transpose |
| `Expected input batch_size (X) to match target batch_size (Y)` | Wrong flatten before `cross_entropy` |
| `index out of range in self` | Token id ≥ vocab size, or position ≥ context_length |
| `mat1 and mat2 shapes cannot be multiplied` | `d_in` ≠ the layer's expected input |
| `The size of tensor a must match tensor b at dimension N` | Broadcasting failure — print both shapes |

**When stuck: print the shape at every line.** It is faster than reasoning about it.
