# Debugging

Symptom → likely cause. Ordered by how often it is actually the answer.

---

## Training

| Symptom | Likely cause |
|---|---|
| Loss is `nan` | Learning rate too high; or a `log(0)` — use fused `cross_entropy` |
| Loss does not decrease at all | Forgot `optimizer.zero_grad()`; frozen parameters; lr ~0 |
| Loss decreases then explodes | No gradient clipping; no warmup |
| Loss suspiciously low from step 1 | **Target leakage** — off-by-one in the shift, or no causal mask |
| Train loss falls, val loss rises | Overfitting. More data, regularization, early stopping |
| Both losses plateau high | Underfitting. Bigger model, better features, less regularization |
| Trains fine, terrible in production | **Data mismatch.** Fix the data, not the model |
| Out of memory | Activations, not weights → reduce batch size first |
| Much slower than expected | Missing `prefetch`/`num_workers`; a silent CPU fallback |
| Different result every run | Set seeds; note that GPU float ops are not associative |

---

## Transformers specifically

| Symptom | Cause |
|---|---|
| Great training loss, garbage generation | **Missing causal mask** — it learned to read the answer |
| Attention weights do not sum to 1 | Masked *after* softmax instead of before with `-inf` |
| Shape error after `.transpose()` | Need `.contiguous()` before `.view()`, or use `.reshape()` |
| Loaded GPT-2 weights produce nonsense | Fused QKV not split; transposes; **LayerNorm `unbiased=True`** |
| Index error at long sequences | Exceeded `context_length` — no positional embedding exists |
| Model ignores word order | No positional encoding. Attention is permutation-invariant |
| Trains badly at depth | Post-norm instead of pre-norm; missing residual connections |
| Gradients vanish | Missing residuals; wrong init for the activation |
| Generation degrades randomly | Forgot `model.eval()` — dropout is still on |

---

## RAG

| Symptom | First thing to check |
|---|---|
| Wrong answers | **Measure retrieval separately.** It is usually there. |
| Cannot find exact IDs or codes | Dense-only. Add BM25 + RRF |
| Retrieval quality mysteriously poor | Missing `"query: "` / `"passage: "` prefix |
| Answer exists in the corpus but is never retrieved | Chunk boundary split it — add overlap |
| Retrieves relevant docs, still answers badly | Too many chunks (distraction); lost-in-the-middle |
| Answers from parametric knowledge | No explicit "use only the context" instruction |
| Confabulates when nothing is retrieved | No "say you don't know" escape hatch |
| Everything broke after a model change | Re-embedded with a different model than the index |

---

## Agents

| Symptom | Cause |
|---|---|
| Runs forever / huge bill | No step cap, no cost budget |
| Repeats the same call | No loop detection. Tell it what it repeated |
| Crashes on a tool failure | Raising instead of returning the error as an observation |
| Picks the wrong tool | Vague description; too many tools (>~12); temperature > 0 |
| Malformed tool arguments | Free-form strings where an enum would do |
| Correct but slow and expensive | **Efficiency failure.** Only reference trajectories reveal it |
| Did something it should not | Tool scoping too broad. Least privilege |
| Behaves oddly after retrieval | **Indirect prompt injection** in a retrieved document |
| Cannot reproduce a failure | Incomplete traces — log tool results too |

---

## LLM output

| Symptom | Cause |
|---|---|
| Different answer each time | You are sampling. `temperature=0` |
| Ignores the output format | Ask nicely → few-shot → **constrained decoding** |
| Cannot count letters / reverse strings | Tokenization. Not fixable by prompting |
| Arithmetic wrong on long numbers | Tokenization of digits. Give it a calculator |
| Costs more than expected in some languages | Tokenizer penalty — 3–5× on non-English |
| Quality dropped with no code change | Provider changed the model. Regression-test continuously |
| Ignores instructions at the end of a long prompt | Lost-in-the-middle. Move them to the edges |
| Prompt works, then breaks after an edit | Whitespace changed the tokenization |

---

## Evaluation

| Symptom | Cause |
|---|---|
| Everything scores 95% | Your eval set is too easy, or contaminated |
| Judge scores disagree with your own | **Validate the judge.** Hand-rate 30 and correlate |
| Scores flip when you reorder options | Position bias. Evaluate both orders |
| Longer answers always win | Verbosity bias. Control for length |
| Improvement is not reproducible | Eval set too small for the effect size |
| Great aggregate, users complain | **Slice it** — by segment, language, difficulty |
| Benchmark score does not predict production | It never did. Build a private eval set |

---

## The general method

1. **Localize before fixing.** Which component? RAG failures are retrieval
   failures far more often than generation failures.
2. **Look at the data.** Read 100 examples. It beats any summary statistic.
3. **Get a baseline.** You cannot know something helped without a before.
4. **Change one thing.** Then measure.
5. **Check the boring causes first.** Off-by-one, wrong mode, wrong dtype, missing
   `zero_grad`. It is almost never exotic.
