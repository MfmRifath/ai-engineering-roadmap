# Notebooks

One runnable notebook per phase. Notebooks are for **exploration** — the moment
something works, move it into [`src/aieng/`](../src/aieng/) where it can be tested.

| Notebook | Phase | Purpose |
|---|---|---|
| `01-ml-foundations.ipynb` | 1 | Géron ch2 end to end on your own dataset |
| `02-deep-learning.ipynb` | 2 | Training loops, the LR range test, the shortcut-gradient experiment |
| `03-llm-intuition.ipynb` | 4 | Tokenizer comparison, attention heatmaps, embedding neighbours |
| `04-transformers.ipynb` | 5 | Building attention four times, shapes printed at every step |
| `05-rag.ipynb` | 6 | The chunk-size × overlap sweep against recall@5 |
| `06-evaluation.ipynb` | 7 | Judge validation against your own ratings |
| `07-agents.ipynb` | 8 | Trajectory traces and failure injection |

## Hygiene

Notebooks are gitignored *outputs*, not gitignored files — install `nbstripout` so
outputs never land in git:

```bash
pip install -e ".[dev]"
nbstripout --install
```

Committed notebook outputs make diffs unreadable and can leak API keys printed in a
cell. Ruff excludes this directory; the code that matters lives in `src/`.
