"""Coding challenges — implement the curriculum's ideas, checked by assertions.

Every challenge is a function you write and a set of tests that either pass or
do not. No model grades anything: the checks are ordinary Python assertions, so
the feedback is deterministic, instant, and identical on every machine.

The set is chosen to cover what the roadmap actually teaches, in roadmap order,
and to be solvable in a few minutes each — these are *comprehension* checks, not
LeetCode. Several are deliberately the same function you will later find in
``aieng``, so you write it once here and recognise it there.

Tests are plain source strings executed after your solution. ``hidden`` tests
are not shown before you run, so the obvious "return the expected value" cheat
does not work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Test:
    name: str
    code: str
    hidden: bool = False


@dataclass(frozen=True)
class Challenge:
    id: str
    title: str
    phase: int
    book: str  # "G4", "R3", ...
    difficulty: int  # 1-5
    prompt: str
    starter: str
    tests: list[Test]
    hints: list[str] = field(default_factory=list)
    solution: str = ""

    @property
    def visible_tests(self) -> list[Test]:
        return [t for t in self.tests if not t.hidden]


CHALLENGES: list[Challenge] = [
    # ── Phase 1 — Géron ────────────────────────────────────────────────
    Challenge(
        id="softmax",
        title="Numerically stable softmax",
        phase=1,
        book="G4",
        difficulty=2,
        prompt=(
            "Implement `softmax(logits)` returning a list of probabilities that sum to 1.\n\n"
            "The naive `exp(x) / sum(exp(x))` overflows for large logits — `exp(1000)` is "
            "`inf`, and `inf/inf` is `nan`. The fix is one line: subtract the maximum "
            "before exponentiating. It changes nothing mathematically, because "
            "`exp(x-c)/sum(exp(x-c))` equals `exp(x)/sum(exp(x))`.\n\n"
            "This is the function at the end of every LLM."
        ),
        starter='import math\n\n\ndef softmax(logits):\n    """Return probabilities that sum to 1."""\n    ...\n',
        tests=[
            Test("sums to 1", "p = softmax([1.0, 2.0, 3.0])\nassert abs(sum(p) - 1.0) < 1e-9"),
            Test(
                "order is preserved",
                "p = softmax([1.0, 2.0, 3.0])\nassert p[2] > p[1] > p[0]",
            ),
            Test(
                "uniform input gives uniform output",
                "p = softmax([5.0, 5.0, 5.0, 5.0])\nassert all(abs(x - 0.25) < 1e-9 for x in p)",
            ),
            Test(
                "does not overflow on large logits",
                "p = softmax([1000.0, 1001.0, 1002.0])\n"
                "assert all(x == x for x in p), 'got nan - subtract the max first'\n"
                "assert abs(sum(p) - 1.0) < 1e-9",
                hidden=True,
            ),
            Test(
                "handles large negatives",
                "p = softmax([-1000.0, -1001.0])\nassert abs(sum(p) - 1.0) < 1e-9",
                hidden=True,
            ),
        ],
        hints=[
            "Subtract max(logits) from every logit before calling exp().",
            "exps = [math.exp(x - m) for x in logits]; then divide by sum(exps).",
        ],
        solution=(
            "import math\n\n\ndef softmax(logits):\n"
            "    m = max(logits)\n"
            "    exps = [math.exp(x - m) for x in logits]\n"
            "    total = sum(exps)\n"
            "    return [e / total for e in exps]\n"
        ),
    ),
    Challenge(
        id="precision_recall",
        title="Precision and recall",
        phase=1,
        book="G3",
        difficulty=1,
        prompt=(
            "Given `y_true` and `y_pred` as lists of 0/1, return `(precision, recall)`.\n\n"
            "precision = TP / (TP + FP) — of what I flagged, how much was right.\n"
            "recall    = TP / (TP + FN) — of what I should have flagged, how much I caught.\n\n"
            "Return 0.0 rather than dividing by zero when the denominator is empty."
        ),
        starter="def precision_recall(y_true, y_pred):\n    ...\n",
        tests=[
            Test(
                "perfect prediction",
                "p, r = precision_recall([1, 0, 1], [1, 0, 1])\nassert p == 1.0 and r == 1.0",
            ),
            Test(
                "one false positive",
                "p, r = precision_recall([1, 0, 0], [1, 1, 0])\n"
                "assert abs(p - 0.5) < 1e-9 and r == 1.0",
            ),
            Test(
                "one false negative",
                "p, r = precision_recall([1, 1, 0], [1, 0, 0])\n"
                "assert p == 1.0 and abs(r - 0.5) < 1e-9",
            ),
            Test(
                "predicting nothing must not divide by zero",
                "p, r = precision_recall([1, 1, 0], [0, 0, 0])\nassert p == 0.0 and r == 0.0",
                hidden=True,
            ),
        ],
        hints=["TP = both 1. FP = predicted 1, actually 0. FN = predicted 0, actually 1."],
        solution=(
            "def precision_recall(y_true, y_pred):\n"
            "    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)\n"
            "    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)\n"
            "    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)\n"
            "    precision = tp / (tp + fp) if tp + fp else 0.0\n"
            "    recall = tp / (tp + fn) if tp + fn else 0.0\n"
            "    return precision, recall\n"
        ),
    ),
    Challenge(
        id="cross_entropy",
        title="Cross-entropy loss",
        phase=1,
        book="G4",
        difficulty=2,
        prompt=(
            "Implement `cross_entropy(probs, target)` where `probs` are predicted "
            "probabilities over classes and `target` is the correct class index.\n\n"
            "Loss = −log(probs[target]).\n\n"
            "Clamp the probability to a small epsilon first: log(0) is −inf, and one "
            "impossible-but-correct prediction would poison the whole batch average.\n\n"
            "This is the loss that trains every LLM."
        ),
        starter="import math\n\nEPS = 1e-12\n\n\ndef cross_entropy(probs, target):\n    ...\n",
        tests=[
            Test(
                "confident and correct is near zero",
                "assert cross_entropy([0.99, 0.01], 0) < 0.02",
            ),
            Test(
                "uniform over 2 classes is ln(2)",
                "assert abs(cross_entropy([0.5, 0.5], 0) - math.log(2)) < 1e-9",
            ),
            Test(
                "confidently wrong is heavily punished",
                "assert cross_entropy([0.001, 0.999], 0) > 6",
            ),
            Test(
                "probability of zero must not return inf",
                "v = cross_entropy([0.0, 1.0], 0)\n"
                "assert v == v and v != float('inf'), 'clamp with EPS before log'",
                hidden=True,
            ),
        ],
        hints=["-math.log(max(probs[target], EPS))"],
        solution=(
            "import math\n\nEPS = 1e-12\n\n\n"
            "def cross_entropy(probs, target):\n"
            "    return -math.log(max(probs[target], EPS))\n"
        ),
    ),
    Challenge(
        id="hash_split",
        title="A train/test split that survives new data",
        phase=1,
        book="G2",
        difficulty=2,
        prompt=(
            "Implement `in_test_set(identifier, test_ratio)` returning True if that row "
            "belongs in the test set.\n\n"
            "A seeded `train_test_split` reshuffles when the dataset grows, so yesterday's "
            "test rows leak into today's training set — silently invalidating every result "
            "you have reported. Hashing a stable id fixes it: each row's side depends only "
            "on itself.\n\n"
            "Use `zlib.crc32` on the identifier's bytes and compare against "
            "`test_ratio * 2**32`."
        ),
        starter=("from zlib import crc32\n\n\ndef in_test_set(identifier, test_ratio):\n    ...\n"),
        tests=[
            Test(
                "deterministic",
                "assert in_test_set(42, 0.2) == in_test_set(42, 0.2)",
            ),
            Test(
                "roughly the right proportion",
                "n = sum(in_test_set(i, 0.2) for i in range(10000))\n"
                "assert 1800 < n < 2200, f'got {n}, expected ~2000'",
            ),
            Test(
                "stable as the dataset grows",
                "small = {i for i in range(1000) if in_test_set(i, 0.2)}\n"
                "large = {i for i in range(5000) if in_test_set(i, 0.2)}\n"
                "assert small == {i for i in large if i < 1000}",
                hidden=True,
            ),
            Test(
                "ratio 0 selects nothing",
                "assert not any(in_test_set(i, 0.0) for i in range(500))",
                hidden=True,
            ),
        ],
        hints=["crc32(str(identifier).encode()) & 0xFFFFFFFF < test_ratio * 2**32"],
        solution=(
            "from zlib import crc32\n\n\n"
            "def in_test_set(identifier, test_ratio):\n"
            "    return crc32(str(identifier).encode()) & 0xFFFFFFFF < test_ratio * 2**32\n"
        ),
    ),
    # ── Phase 4-5 — Alammar / Raschka ─────────────────────────────────
    Challenge(
        id="bpe_merge",
        title="One BPE merge step",
        phase=5,
        book="R2",
        difficulty=3,
        prompt=(
            "Implement `merge(ids, pair, new_id)`: replace every non-overlapping "
            "occurrence of the adjacent `pair` in the list `ids` with `new_id`.\n\n"
            "This is the inner loop of byte pair encoding. Repeat it a few thousand times "
            "and you have a tokenizer.\n\n"
            "Careful with overlaps: merging (1,1) in [1,1,1] gives [new,1], not [new,new]."
        ),
        starter="def merge(ids, pair, new_id):\n    ...\n",
        tests=[
            Test(
                "single merge",
                "assert merge([1, 2, 3], (1, 2), 99) == [99, 3]",
            ),
            Test(
                "multiple merges",
                "assert merge([1, 2, 1, 2], (1, 2), 99) == [99, 99]",
            ),
            Test(
                "no match leaves it unchanged",
                "assert merge([1, 2, 3], (5, 6), 99) == [1, 2, 3]",
            ),
            Test(
                "overlapping pairs merge left to right",
                "assert merge([1, 1, 1], (1, 1), 99) == [99, 1]",
                hidden=True,
            ),
            Test(
                "short and empty inputs",
                "assert merge([], (1, 2), 99) == []\nassert merge([1], (1, 2), 99) == [1]",
                hidden=True,
            ),
        ],
        hints=[
            "Walk with an index while i < len(ids).",
            "On a match append new_id and i += 2, otherwise append ids[i] and i += 1.",
        ],
        solution=(
            "def merge(ids, pair, new_id):\n"
            "    out, i = [], 0\n"
            "    while i < len(ids):\n"
            "        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:\n"
            "            out.append(new_id)\n"
            "            i += 2\n"
            "        else:\n"
            "            out.append(ids[i])\n"
            "            i += 1\n"
            "    return out\n"
        ),
    ),
    Challenge(
        id="sliding_windows",
        title="Next-token training pairs",
        phase=5,
        book="R2",
        difficulty=2,
        prompt=(
            "Implement `windows(tokens, max_length, stride)` returning a list of "
            "`(input, target)` tuples, where target is input shifted right by one.\n\n"
            "Getting this off by one is the classic chapter-2 bug: the model learns to "
            "copy its input, which shows up as a suspiciously low loss.\n\n"
            "Emit a window only when a full target of `max_length` exists."
        ),
        starter="def windows(tokens, max_length, stride):\n    ...\n",
        tests=[
            Test(
                "target is shifted by one",
                "w = windows([1, 2, 3, 4, 5], 2, 2)\nassert w[0] == ([1, 2], [2, 3])",
            ),
            Test(
                "stride controls the step",
                "w = windows([1, 2, 3, 4, 5], 2, 2)\nassert w[1] == ([3, 4], [4, 5])",
            ),
            Test(
                "no truncated window at the end",
                "for inp, tgt in windows(list(range(10)), 3, 3):\n"
                "    assert len(inp) == 3 and len(tgt) == 3",
                hidden=True,
            ),
            Test(
                "too short for one window",
                "assert windows([1, 2], 5, 1) == []",
                hidden=True,
            ),
        ],
        hints=["for i in range(0, len(tokens) - max_length, stride)"],
        solution=(
            "def windows(tokens, max_length, stride):\n"
            "    out = []\n"
            "    for i in range(0, len(tokens) - max_length, stride):\n"
            "        out.append((tokens[i : i + max_length], tokens[i + 1 : i + max_length + 1]))\n"
            "    return out\n"
        ),
    ),
    Challenge(
        id="causal_mask",
        title="The causal mask",
        phase=5,
        book="R3",
        difficulty=2,
        prompt=(
            "Implement `causal_mask(n)` returning an n x n list of lists where "
            "`mask[i][j]` is True when position j must be **hidden** from position i.\n\n"
            "A position may attend to itself and everything before it, never ahead. "
            "So the strict upper triangle is masked."
        ),
        starter="def causal_mask(n):\n    ...\n",
        tests=[
            Test(
                "3x3 shape",
                "m = causal_mask(3)\nassert len(m) == 3 and all(len(r) == 3 for r in m)",
            ),
            Test(
                "first row hides everything after position 0",
                "m = causal_mask(3)\nassert m[0] == [False, True, True]",
            ),
            Test(
                "last row hides nothing",
                "m = causal_mask(3)\nassert m[2] == [False, False, False]",
            ),
            Test(
                "diagonal is always visible",
                "m = causal_mask(6)\nassert all(m[i][i] is False for i in range(6))",
                hidden=True,
            ),
            Test(
                "exactly n*(n-1)/2 masked cells",
                "m = causal_mask(5)\nassert sum(sum(r) for r in m) == 10",
                hidden=True,
            ),
        ],
        hints=["mask[i][j] is True when j > i."],
        solution=(
            "def causal_mask(n):\n    return [[j > i for j in range(n)] for i in range(n)]\n"
        ),
    ),
    Challenge(
        id="attention",
        title="Scaled dot-product attention",
        phase=5,
        book="R3",
        difficulty=4,
        prompt=(
            "Implement `attention(Q, K, V)` with numpy, returning the output matrix.\n\n"
            "    softmax(QKᵀ / √d_k) · V\n\n"
            "Three steps: score every query against every key, normalise the scores into "
            "weights that sum to 1 per row, take the weighted average of the values.\n\n"
            "The √d_k matters — without it, large dot products saturate the softmax and "
            "gradients vanish."
        ),
        starter=(
            "import numpy as np\n\n\n"
            "def attention(Q, K, V):\n"
            '    """Q, K, V are (seq_len, d). Return (seq_len, d_v)."""\n'
            "    ...\n"
        ),
        tests=[
            Test(
                "output shape",
                "import numpy as np\n"
                "Q = np.random.randn(4, 8); K = np.random.randn(4, 8); V = np.random.randn(4, 16)\n"
                "assert attention(Q, K, V).shape == (4, 16)",
            ),
            Test(
                "identical keys give the mean of the values",
                "import numpy as np\n"
                "Q = np.zeros((1, 4)); K = np.zeros((3, 4)); V = np.array([[1.0], [2.0], [3.0]])\n"
                "assert abs(attention(Q, K, V)[0, 0] - 2.0) < 1e-6",
            ),
            Test(
                "attention weights sum to 1",
                "import numpy as np\n"
                "Q = np.random.randn(3, 8); K = np.random.randn(5, 8); V = np.eye(5)\n"
                "out = attention(Q, K, V)\n"
                "assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)",
                hidden=True,
            ),
            Test(
                "the sqrt(d_k) scaling is applied",
                "import numpy as np\n"
                "Q = np.array([[10.0, 0.0, 0.0, 0.0]]); K = np.eye(4) * 10\n"
                "V = np.arange(4, dtype=float).reshape(4, 1)\n"
                "got = attention(Q, K, V)[0, 0]\n"
                "s = np.array([100.0, 0, 0, 0]) / 2.0\n"
                "w = np.exp(s - s.max()); w /= w.sum()\n"
                "assert abs(got - float(w @ np.arange(4))) < 1e-6, 'divide scores by sqrt(d_k)'",
                hidden=True,
            ),
        ],
        hints=[
            "scores = Q @ K.T / np.sqrt(Q.shape[-1])",
            "Subtract scores.max(axis=-1, keepdims=True) before exp for stability.",
            "weights = exps / exps.sum(axis=-1, keepdims=True); return weights @ V",
        ],
        solution=(
            "import numpy as np\n\n\n"
            "def attention(Q, K, V):\n"
            "    scores = Q @ K.T / np.sqrt(Q.shape[-1])\n"
            "    scores = scores - scores.max(axis=-1, keepdims=True)\n"
            "    weights = np.exp(scores)\n"
            "    weights /= weights.sum(axis=-1, keepdims=True)\n"
            "    return weights @ V\n"
        ),
    ),
    Challenge(
        id="temperature_topk",
        title="Temperature and top-k sampling",
        phase=5,
        book="R5",
        difficulty=3,
        prompt=(
            "Implement `apply_sampling(logits, temperature, top_k)` returning the "
            "**probabilities** after applying top-k filtering and temperature.\n\n"
            "Order matters: keep the top-k logits (set the rest to −inf), divide by "
            "temperature, then softmax. Setting rejected logits to −inf before the softmax "
            "is the same trick as the causal mask — `exp(-inf)` is 0, so the survivors "
            "still normalise to 1.\n\n"
            "Temperature divides the **logits**, never the probabilities."
        ),
        starter=(
            "import math\n\n\ndef apply_sampling(logits, temperature=1.0, top_k=None):\n    ...\n"
        ),
        tests=[
            Test(
                "still sums to 1",
                "p = apply_sampling([1.0, 2.0, 3.0], 1.0, None)\nassert abs(sum(p) - 1.0) < 1e-9",
            ),
            Test(
                "top_k zeroes everything outside the top k",
                "p = apply_sampling([1.0, 2.0, 3.0, 4.0], 1.0, 2)\n"
                "assert p[0] == 0.0 and p[1] == 0.0\n"
                "assert abs(p[2] + p[3] - 1.0) < 1e-9",
            ),
            Test(
                "low temperature sharpens toward the argmax",
                "p = apply_sampling([1.0, 2.0, 3.0], 0.1, None)\nassert p[2] > 0.99",
            ),
            Test(
                "high temperature flattens toward uniform",
                "p = apply_sampling([1.0, 2.0, 3.0], 100.0, None)\nassert max(p) - min(p) < 0.05",
                hidden=True,
            ),
            Test(
                "top_k larger than the vocabulary is harmless",
                "p = apply_sampling([1.0, 2.0], 1.0, 10)\nassert abs(sum(p) - 1.0) < 1e-9",
                hidden=True,
            ),
        ],
        hints=[
            "Find the k-th largest logit, then replace anything below it with -inf.",
            "Divide by temperature after filtering, then softmax with the max subtracted.",
        ],
        solution=(
            "import math\n\n\n"
            "def apply_sampling(logits, temperature=1.0, top_k=None):\n"
            "    xs = list(logits)\n"
            "    if top_k:\n"
            "        k = min(top_k, len(xs))\n"
            "        cutoff = sorted(xs, reverse=True)[k - 1]\n"
            "        xs = [x if x >= cutoff else -math.inf for x in xs]\n"
            "    xs = [x / temperature for x in xs]\n"
            "    m = max(xs)\n"
            "    exps = [0.0 if x == -math.inf else math.exp(x - m) for x in xs]\n"
            "    total = sum(exps)\n"
            "    return [e / total for e in exps]\n"
        ),
    ),
    Challenge(
        id="layer_norm",
        title="Layer normalization",
        phase=5,
        book="R4",
        difficulty=3,
        prompt=(
            "Implement `layer_norm(x, eps=1e-5)` for a single vector: subtract the mean, "
            "divide by the standard deviation.\n\n"
            "Use the **biased** variance (divide by n, not n−1) — that is what GPT-2 does, "
            "and using the unbiased estimator makes loaded OpenAI weights produce subtly "
            "wrong outputs.\n\n"
            "Note this normalises across features within one example, which is why it works "
            "at batch size 1 and BatchNorm does not."
        ),
        starter="def layer_norm(x, eps=1e-5):\n    ...\n",
        tests=[
            Test(
                "mean becomes ~0",
                "out = layer_norm([1.0, 2.0, 3.0, 4.0])\nassert abs(sum(out) / len(out)) < 1e-6",
            ),
            Test(
                "std becomes ~1",
                "out = layer_norm([1.0, 2.0, 3.0, 4.0])\n"
                "m = sum(out) / len(out)\n"
                "var = sum((v - m) ** 2 for v in out) / len(out)\n"
                "assert abs(var - 1.0) < 1e-3",
            ),
            Test(
                "constant input does not divide by zero",
                "out = layer_norm([5.0, 5.0, 5.0])\nassert all(v == v for v in out)",
                hidden=True,
            ),
            Test(
                "uses biased variance",
                "out = layer_norm([0.0, 2.0])\nassert abs(abs(out[0]) - 1.0) < 1e-3",
                hidden=True,
            ),
        ],
        hints=["var = sum((v - mean) ** 2 for v in x) / len(x)  # divide by n"],
        solution=(
            "def layer_norm(x, eps=1e-5):\n"
            "    mean = sum(x) / len(x)\n"
            "    var = sum((v - mean) ** 2 for v in x) / len(x)\n"
            "    denom = (var + eps) ** 0.5\n"
            "    return [(v - mean) / denom for v in x]\n"
        ),
    ),
    # ── Phase 6-7 — Huyen ─────────────────────────────────────────────
    Challenge(
        id="kv_cache_bytes",
        title="KV cache memory",
        phase=7,
        book="H9",
        difficulty=2,
        prompt=(
            "Implement `kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, "
            "bytes_per=2)`.\n\n"
            "    2 × layers × kv_heads × head_dim × seq_len × batch × bytes / 1e9\n\n"
            "The 2 is keys **and** values. Note `n_kv_heads`, not `n_heads` — with "
            "grouped-query attention several query heads share one KV head, and shrinking "
            "this tensor is the entire point of GQA.\n\n"
            "At long context this exceeds the model weights and becomes what limits "
            "concurrency."
        ),
        starter=(
            "def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per=2):\n"
            "    ...\n"
        ),
        tests=[
            Test(
                "Llama-3-8B at 8k context, batch 1",
                "v = kv_cache_gb(32, 8, 128, 8192, 1)\nassert abs(v - 1.0737) < 0.01",
            ),
            Test(
                "linear in batch size",
                "a = kv_cache_gb(32, 8, 128, 4096, 1)\n"
                "b = kv_cache_gb(32, 8, 128, 4096, 8)\n"
                "assert abs(b - 8 * a) < 1e-6",
            ),
            Test(
                "GQA with 4x fewer KV heads is 4x smaller",
                "mha = kv_cache_gb(32, 32, 128, 8192, 1)\n"
                "gqa = kv_cache_gb(32, 8, 128, 8192, 1)\n"
                "assert abs(gqa - mha / 4) < 1e-6",
                hidden=True,
            ),
            Test(
                "exceeds 16 GB of weights at batch 32",
                "assert kv_cache_gb(32, 8, 128, 8192, 32) > 16",
                hidden=True,
            ),
        ],
        hints=["Do not forget the leading factor of 2 for keys and values."],
        solution=(
            "def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per=2):\n"
            "    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per / 1e9\n"
        ),
    ),
    Challenge(
        id="recall_at_k",
        title="recall@k",
        phase=7,
        book="H4",
        difficulty=1,
        prompt=(
            "Implement `recall_at_k(retrieved, relevant, k)` — of the relevant documents, "
            "what fraction appeared in the top k.\n\n"
            "`retrieved` is a ranked list of ids, `relevant` a set of ids.\n\n"
            "This is the metric that matters most for RAG: retrieval quality is the ceiling "
            "on system quality, because the generator cannot answer from a passage it never "
            "received.\n\n"
            "With nothing relevant, return 1.0 — vacuously satisfied."
        ),
        starter="def recall_at_k(retrieved, relevant, k):\n    ...\n",
        tests=[
            Test(
                "three of four found in the top 5",
                "assert abs(recall_at_k(['a','x','b','y','c'], {'a','b','c','d'}, 5) - 0.75) < 1e-9",
            ),
            Test(
                "cut off at k",
                "assert abs(recall_at_k(['a','x','b'], {'a','b'}, 1) - 0.5) < 1e-9",
            ),
            Test(
                "indifferent to order within k",
                "assert recall_at_k(['a','b','x'], {'a','b'}, 3) == recall_at_k(['x','b','a'], {'a','b'}, 3)",
                hidden=True,
            ),
            Test(
                "no relevant documents is vacuously 1.0",
                "assert recall_at_k(['a'], set(), 5) == 1.0",
                hidden=True,
            ),
        ],
        hints=["len(set(retrieved[:k]) & relevant) / len(relevant)"],
        solution=(
            "def recall_at_k(retrieved, relevant, k):\n"
            "    if not relevant:\n"
            "        return 1.0\n"
            "    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)\n"
        ),
    ),
    Challenge(
        id="rrf",
        title="Reciprocal rank fusion",
        phase=7,
        book="H6",
        difficulty=3,
        prompt=(
            "Implement `rrf(rankings, k=60)` returning document ids sorted best-first.\n\n"
            "Each document scores `sum(1 / (k + rank))` over the rankings it appears in, "
            "with rank counted from 1.\n\n"
            "This is how hybrid retrieval combines BM25 with dense search. Because only "
            "**ranks** are used, the two systems' incomparable score scales never have to be "
            "normalised — which is exactly what makes naive score-blending fragile."
        ),
        starter="def rrf(rankings, k=60):\n    ...\n",
        tests=[
            Test(
                "ranked well by both retrievers wins",
                "out = rrf([['a','b','c'], ['c','a','b']])\nassert out[0] == 'a'",
            ),
            Test(
                "keeps documents found by only one retriever",
                "assert set(rrf([['a','b'], ['c','d']])) == {'a','b','c','d'}",
            ),
            Test(
                "a single ranking is preserved",
                "assert rrf([['x','y','z']]) == ['x','y','z']",
                hidden=True,
            ),
            Test(
                "empty input",
                "assert rrf([]) == []",
                hidden=True,
            ),
        ],
        hints=["scores[doc] += 1 / (k + rank + 1) with rank from enumerate() starting at 0."],
        solution=(
            "def rrf(rankings, k=60):\n"
            "    scores = {}\n"
            "    for ranking in rankings:\n"
            "        for rank, doc in enumerate(ranking):\n"
            "            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)\n"
            "    return sorted(scores, key=lambda d: (-scores[d], d))\n"
        ),
    ),
    Challenge(
        id="chunk_overlap",
        title="Chunking with overlap",
        phase=7,
        book="A8",
        difficulty=3,
        prompt=(
            "Implement `chunk(text, size, overlap)` returning a list of strings.\n\n"
            "Every chunk must be at most `size` characters, consecutive chunks must share "
            "`overlap` characters, and the whole text must be covered.\n\n"
            "Overlap is what stops an answer that straddles a boundary from becoming "
            "unretrievable — it would appear in neither chunk completely."
        ),
        starter="def chunk(text, size, overlap):\n    ...\n",
        tests=[
            Test(
                "respects the size bound",
                "assert all(len(c) <= 10 for c in chunk('a' * 45, 10, 3))",
            ),
            Test(
                "short text is a single chunk",
                "assert chunk('hello', 100, 10) == ['hello']",
            ),
            Test(
                "consecutive chunks overlap",
                "cs = chunk('abcdefghijklmnop', 8, 3)\nassert cs[0][-3:] == cs[1][:3]",
            ),
            Test(
                "covers the whole text",
                "t = ''.join(chr(97 + i % 26) for i in range(100))\n"
                "cs = chunk(t, 12, 4)\n"
                "assert cs[0][0] == t[0] and cs[-1][-1] == t[-1]",
                hidden=True,
            ),
            Test(
                "empty text",
                "assert chunk('', 10, 2) == []",
                hidden=True,
            ),
        ],
        hints=["step = size - overlap; start at 0 and advance by step."],
        solution=(
            "def chunk(text, size, overlap):\n"
            "    if not text:\n"
            "        return []\n"
            "    if len(text) <= size:\n"
            "        return [text]\n"
            "    step = size - overlap\n"
            "    out, start = [], 0\n"
            "    while start < len(text):\n"
            "        out.append(text[start : start + size])\n"
            "        if start + size >= len(text):\n"
            "            break\n"
            "        start += step\n"
            "    return out\n"
        ),
    ),
    Challenge(
        id="training_memory",
        title="Fine-tuning memory budget",
        phase=6,
        book="H7",
        difficulty=3,
        prompt=(
            "Implement `training_gb(n_params_b, bytes_per_param=2, optimizer_bytes=8, "
            "trainable_fraction=1.0)` returning total GB (ignoring activations).\n\n"
            "  weights   = all params × bytes_per_param\n"
            "  gradients = trainable params × bytes_per_param\n"
            "  optimizer = trainable params × optimizer_bytes\n\n"
            "AdamW keeps two fp32 moments per **trainable** parameter — 8 bytes. That is why "
            "full fine-tuning a 7B needs ~84 GB, and why LoRA, by making the trainable "
            "fraction ~0.5%, makes the same job fit on a laptop GPU."
        ),
        starter=(
            "def training_gb(n_params_b, bytes_per_param=2, optimizer_bytes=8, "
            "trainable_fraction=1.0):\n    ...\n"
        ),
        tests=[
            Test(
                "full fine-tune of a 7B is ~84 GB",
                "assert abs(training_gb(7) - 84.0) < 0.5",
            ),
            Test(
                "Adam state alone is 56 GB",
                "full = training_gb(7)\n"
                "no_opt = training_gb(7, optimizer_bytes=0)\n"
                "assert abs((full - no_opt) - 56.0) < 0.5",
            ),
            Test(
                "LoRA at 0.5% trainable fits in 16 GB",
                "assert training_gb(7, trainable_fraction=0.005) < 16",
                hidden=True,
            ),
            Test(
                "QLoRA: 4-bit base plus LoRA is under 5 GB",
                "assert training_gb(7, bytes_per_param=0.5, trainable_fraction=0.005) < 5",
                hidden=True,
            ),
        ],
        hints=["trainable = n_params_b * 1e9 * trainable_fraction"],
        solution=(
            "def training_gb(n_params_b, bytes_per_param=2, optimizer_bytes=8, "
            "trainable_fraction=1.0):\n"
            "    params = n_params_b * 1e9\n"
            "    trainable = params * trainable_fraction\n"
            "    total = params * bytes_per_param + trainable * bytes_per_param"
            " + trainable * optimizer_bytes\n"
            "    return total / 1e9\n"
        ),
    ),
    Challenge(
        id="compounding",
        title="Compounding reliability",
        phase=8,
        book="H6",
        difficulty=1,
        prompt=(
            "Implement `end_to_end(per_step, steps)` — the probability that every step in "
            "a sequence succeeds.\n\n"
            "Success rates multiply. This one line is the constraint on every multi-step "
            "LLM system, and the reason most production 'agents' are constrained workflows "
            "with one or two model decision points."
        ),
        starter="def end_to_end(per_step, steps):\n    ...\n",
        tests=[
            Test(
                "95% over 10 steps is ~60%",
                "assert abs(end_to_end(0.95, 10) - 0.5987) < 1e-3",
            ),
            Test(
                "90% over 4 steps is ~66%",
                "assert abs(end_to_end(0.90, 4) - 0.6561) < 1e-3",
            ),
            Test(
                "a single step is itself",
                "assert abs(end_to_end(0.8, 1) - 0.8) < 1e-9",
                hidden=True,
            ),
            Test(
                "perfect stays perfect",
                "assert end_to_end(1.0, 100) == 1.0",
                hidden=True,
            ),
        ],
        hints=["per_step ** steps"],
        solution="def end_to_end(per_step, steps):\n    return per_step**steps\n",
    ),
    Challenge(
        id="loop_detection",
        title="Agent loop detection",
        phase=8,
        book="L6",
        difficulty=3,
        prompt=(
            "Implement `first_repeat(calls)` returning the index of the first tool call "
            "that repeats an earlier one exactly, or `None`.\n\n"
            "Each call is a `(name, args_dict)` tuple. Two calls are the same if the name "
            "and arguments match — **regardless of dict key order**.\n\n"
            "Two identical calls mean the agent did not learn from the first result. This "
            "check is what stands between you and a runaway bill."
        ),
        starter="import json\n\n\ndef first_repeat(calls):\n    ...\n",
        tests=[
            Test(
                "detects an exact repeat",
                "calls = [('search', {'q': 'x'}), ('read', {'id': 1}), ('search', {'q': 'x'})]\n"
                "assert first_repeat(calls) == 2",
            ),
            Test(
                "different arguments are not a repeat",
                "assert first_repeat([('search', {'q': 'a'}), ('search', {'q': 'b'})]) is None",
            ),
            Test(
                "key order must not matter",
                "calls = [('t', {'a': 1, 'b': 2}), ('t', {'b': 2, 'a': 1})]\n"
                "assert first_repeat(calls) == 1",
                hidden=True,
            ),
            Test(
                "empty and single-call inputs",
                "assert first_repeat([]) is None\nassert first_repeat([('a', {})]) is None",
                hidden=True,
            ),
        ],
        hints=["json.dumps(args, sort_keys=True) gives an order-independent key."],
        solution=(
            "import json\n\n\n"
            "def first_repeat(calls):\n"
            "    seen = set()\n"
            "    for i, (name, args) in enumerate(calls):\n"
            "        key = (name, json.dumps(args, sort_keys=True, default=str))\n"
            "        if key in seen:\n"
            "            return i\n"
            "        seen.add(key)\n"
            "    return None\n"
        ),
    ),
    Challenge(
        id="sm2",
        title="SM-2 scheduling",
        phase=8,
        book="—",
        difficulty=4,
        prompt=(
            "Implement `next_interval(repetitions, ease, interval, grade)` returning "
            "`(repetitions, ease, interval)` after one review. Grade is 0–5.\n\n"
            "  new ease = ease + (0.1 − (5−q)(0.08 + (5−q)·0.02)), floored at 1.3\n"
            "  if q < 3: repetitions = 0, interval = 1  (a lapse — but keep the new ease)\n"
            "  else: repetitions += 1; interval = 1, then 6, then round(interval × ease)\n\n"
            "This is the algorithm scheduling the flashcards in this very app."
        ),
        starter="def next_interval(repetitions, ease, interval, grade):\n    ...\n",
        tests=[
            Test(
                "first success is 1 day",
                "r, e, i = next_interval(0, 2.5, 0, 4)\nassert i == 1 and r == 1",
            ),
            Test(
                "second success is 6 days",
                "r, e, i = next_interval(1, 2.5, 1, 4)\nassert i == 6 and r == 2",
            ),
            Test(
                "third multiplies by ease",
                "r, e, i = next_interval(2, 2.5, 6, 4)\nassert i == round(6 * e)",
            ),
            Test(
                "a lapse resets the interval but keeps the lowered ease",
                "r, e, i = next_interval(5, 2.5, 40, 0)\nassert r == 0 and i == 1 and e < 2.5",
                hidden=True,
            ),
            Test(
                "ease never drops below 1.3",
                "e = 2.5\nr, i = 0, 0\n"
                "for _ in range(30):\n"
                "    r, e, i = next_interval(r, e, i, 0)\n"
                "assert abs(e - 1.3) < 1e-9",
                hidden=True,
            ),
        ],
        hints=[
            "Compute the new ease first, and apply it even on a lapse.",
            "max(1.3, ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))",
        ],
        solution=(
            "def next_interval(repetitions, ease, interval, grade):\n"
            "    q = grade\n"
            "    ease = max(1.3, ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))\n"
            "    if q < 3:\n"
            "        return 0, ease, 1\n"
            "    repetitions += 1\n"
            "    if repetitions == 1:\n"
            "        interval = 1\n"
            "    elif repetitions == 2:\n"
            "        interval = 6\n"
            "    else:\n"
            "        interval = max(1, round(interval * ease))\n"
            "    return repetitions, ease, interval\n"
        ),
    ),
]

BY_ID = {c.id: c for c in CHALLENGES}


def get(challenge_id: str) -> Challenge | None:
    return BY_ID.get(challenge_id)


def summary() -> list[dict]:
    return [
        {
            "id": c.id,
            "title": c.title,
            "phase": c.phase,
            "book": c.book,
            "difficulty": c.difficulty,
            "tests": len(c.tests),
        }
        for c in CHALLENGES
    ]
