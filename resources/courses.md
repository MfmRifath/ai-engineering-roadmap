# Courses, videos, and other resources

Things worth your time alongside the books. Deliberately short — a long list is a
way of avoiding starting.

---

## Video, when you want a second pass

- ⭐ **[Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)**
  — the same journey as [Raschka](../books/03-build-llm-from-scratch-raschka/), in
  two hours. Excellent *after* Phase 5; watching it before robs you of the exercise.
- **[Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**
  — the whole series. `micrograd` makes backprop unforgettable.
- **[Karpathy — Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI)**
  — a state-of-the-field overview.
- **[3Blue1Brown — Neural networks](https://www.3blue1brown.com/topics/neural-networks)**
  — the best visual intuition for backprop and, more recently, attention.

## Written explainers

- ⭐ **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
  — Alammar's post. Read alongside
  [A3](../books/02-hands-on-llms-alammar/notes/ch03.md).
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
  — the 2017 paper, line by line, as runnable code.
- **[huyenchip.com/blog](https://huyenchip.com/blog/)** — much of
  [AI Engineering](../books/04-ai-engineering-huyen/) appeared here first, and stays
  updated.
- **[Sebastian Raschka — Ahead of AI](https://magazine.sebastianraschka.com/)** —
  the best regular summary of what actually changed.
- **[Lilian Weng's blog](https://lilianweng.github.io/)** — deep, careful surveys.
  Her agent and hallucination posts are standard references.

## Courses

- **[fast.ai — Practical Deep Learning](https://course.fast.ai/)** — top-down, the
  opposite pedagogy to Géron. Good as a complement.
- **[Stanford CS224N](https://web.stanford.edu/class/cs224n/)** — NLP with deep
  learning. Lectures are public.
- **[Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/)**
  — builds an LLM end to end. The natural next step after Phase 5.
- **[DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/)** —
  an hour each, useful for a specific tool.

## Documentation worth reading properly

- **[PyTorch tutorials](https://pytorch.org/tutorials/)** — start with the 60-minute
  blitz if Appendix A of Raschka was not enough.
- **[Hugging Face course](https://huggingface.co/learn/nlp-course)** — free, and the
  `transformers` library is the durable one to know.
- **[scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)** —
  genuinely excellent prose, not just API reference.
- **[vLLM docs](https://docs.vllm.ai/)** — continuous batching and PagedAttention in
  practice ([H9](../books/04-ai-engineering-huyen/notes/ch09.md)).

## Official book code

- [ageron/handson-ml2](https://github.com/ageron/handson-ml2) ·
  [handson-ml3](https://github.com/ageron/handson-ml3)
- [HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — very well
  maintained, with bonus material beyond the book

## Practice

- **[Kaggle](https://www.kaggle.com/)** — for Phase 1–2. Read winning solutions;
  they teach feature engineering better than any book.
- **[Papers with Code](https://paperswithcode.com/)** — implementations to compare
  against yours.
- **Your own data.** The best project is one you want the answer to.

## Staying current without drowning

The field produces more than anyone can read. A sustainable filter:

1. **One newsletter**, read weekly. Not five, read never.
2. **Primary sources over commentary** for anything you will act on.
3. **Ignore benchmark announcements.** They are marketing, and contaminated.
4. **Follow the durable questions** — evaluation, cost, reliability — rather than
   model releases. The questions outlive the models.

The methodology in these five books will still be true when every model named in
them is retired. That is the bet this roadmap makes.
