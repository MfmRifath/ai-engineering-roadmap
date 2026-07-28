# Projects

Six builds that force you to prove you learned it. Notes are how you remember;
projects are how you find out what you actually understood.

| Project | After phase | Draws on | You end up with |
|---|---|---|---|
| [p1 — End-to-end ML service](p1-end-to-end-ml-service/) | 1 | G2, G19 | A trained model behind an API, containerized |
| [p2 — RAG over my library](p2-rag-over-my-library/) | 4 | A8, H6 | Semantic search over your own PDFs, with citations |
| [p3 — nanoGPT from scratch](p3-nanogpt-from-scratch/) | 5 | R2–R5 | A GPT you wrote and pretrained yourself |
| [p4 — LLM eval harness](p4-llm-eval-harness/) | 7 | H3, H4 | A CLI that scores models and catches regressions |
| [p5 — Agent platform](p5-agent-platform/) | 8 | L5–L11 | A tool-using agent that recovers from failures |
| [p6 — Production LLM app](p6-production-llm-app/) | 8 | H10 + all | The capstone |

## How to work on these

Each project has a `README.md` with a **spec**, a **definition of done**, and
**stretch goals**. None of them ships finished code — that would defeat the point.
What they give you is the target, the pitfalls, and the tests you should be able
to pass.

Reuse [`src/aieng/`](../src/aieng/) rather than rewriting. That package exists so
your chapter work accumulates into something you can build on.

## The rule that matters

**Ship something that runs.** A notebook that produced a number once is not a
project. Each of these should be runnable by someone else — you in six months —
from a README and a command.
