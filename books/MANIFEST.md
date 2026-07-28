# The Library

The five books this curriculum is built on. **None of them are in this repository.**

The notes under `books/*/notes/` are original summaries written while studying — they
paraphrase and connect ideas, they do not reproduce the books. To actually follow this
roadmap you need the books. Buy them; they are worth it, and the authors deserve it.

Put your copies in `library/` at the repo root. That directory is gitignored, so it never
leaves your machine. `make toc` reads them from there to regenerate the chapter maps.

```
library/            <- gitignored, your copies live here
├── geron-hands-on-ml.pdf
├── alammar-hands-on-llms.pdf
├── raschka-build-llm.pdf
├── huyen-ai-engineering.pdf
└── lanham-ai-agents.pdf
```

---

## 1. Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow

| | |
|---|---|
| **Author** | Aurélien Géron |
| **Edition** | 2nd edition, 2019 (a 3rd edition, 2022, is available and preferred if buying new) |
| **Publisher** | O'Reilly Media |
| **ISBN-13** | 978-1492032649 |
| **Chapters** | 19 |
| **Buy** | [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) · [Publisher listing](https://www.oreilly.com/search/?q=hands-on%20machine%20learning) |
| **Free code** | [github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2) |

**Role in the roadmap:** the foundation. Everything downstream assumes you are comfortable
with train/test discipline, regularization, gradient descent, and what overfitting feels
like in practice. Skip this and the LLM material becomes cargo-culting.

> **On the edition:** the 2nd edition uses TensorFlow 1.x-era idioms in places and predates
> the transformer's dominance. The ML fundamentals (ch. 1–9) are timeless. For ch. 10–19,
> read for concepts and expect the API details to have moved on. Notes flag this per chapter.

---

## 2. Hands-On Large Language Models

| | |
|---|---|
| **Authors** | Jay Alammar, Maarten Grootendorst |
| **Edition** | 1st edition, 2024 |
| **Publisher** | O'Reilly Media |
| **ISBN-13** | 978-1098150969 |
| **Chapters** | 12, in 3 parts |
| **Buy** | [O'Reilly](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) |
| **Free code** | [github.com/HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) |
| **Companion** | Jay Alammar's [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |

**Role in the roadmap:** intuition. This is the book that makes you *see* what a transformer
is doing before you are asked to implement one. Its visual explanations of tokenization,
embeddings, and the forward pass are the best in print.

---

## 3. Build a Large Language Model (From Scratch)

| | |
|---|---|
| **Author** | Sebastian Raschka |
| **Edition** | 1st edition, 2024 |
| **Publisher** | Manning Publications |
| **ISBN-13** | 978-1633437166 |
| **Chapters** | 7 + appendices |
| **Buy** | [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch) |
| **Free code** | [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) |

**Role in the roadmap:** the demystification. Seven chapters take you from raw text to a
working GPT you trained yourself, then fine-tune it twice. After this, no part of an LLM is
a black box. This is the highest value-per-page book of the five.

---

## 4. AI Engineering: Building Applications with Foundation Models

| | |
|---|---|
| **Author** | Chip Huyen |
| **Edition** | 1st edition, 2025 |
| **Publisher** | O'Reilly Media |
| **ISBN-13** | 978-1098166304 |
| **Chapters** | 10 |
| **Buy** | [O'Reilly](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) |
| **Companion** | [Chip Huyen's blog](https://huyenchip.com/blog/) · her *Designing Machine Learning Systems* is the sibling volume |

**Role in the roadmap:** the profession. This is the book that separates "I can call an API"
from "I can ship and operate this." Evaluation (ch. 3–4) is the single most under-practiced
skill in the field, and this is the best treatment of it that exists.

---

## 5. AI Agents in Action

| | |
|---|---|
| **Author** | Michael Lanham |
| **Edition** | 1st edition, 2025 |
| **Publisher** | Manning Publications |
| **ISBN-13** | 978-1633436343 |
| **Chapters** | 11 |
| **Buy** | [Manning](https://www.manning.com/books/ai-agents-in-action) |

**Role in the roadmap:** the frontier. Agents are where the field is least settled, so read
this one most critically — the specific frameworks it uses will age faster than anything
else on this list. The durable content is the decomposition: actions, memory, planning,
reasoning, evaluation, and multi-agent coordination as separable concerns.

---

## Reading order

Not the order above — see [ROADMAP.md](../ROADMAP.md). Short version: Géron for
foundations, Alammar for intuition, Raschka to build it, Huyen to ship it, Lanham to
extend it. The books interleave; [CURRICULUM.md](../CURRICULUM.md) maps every concept to
every book that covers it.

## Verifying your copies

`make toc` parses the PDF outlines in `library/` and checks the chapter counts against what
the notes assume: 19 / 12 / 7 / 10 / 11. If your edition differs, the notes may be
one chapter off — the script will tell you.

```bash
make toc            # or: python scripts/extract_toc.py --verify
```
