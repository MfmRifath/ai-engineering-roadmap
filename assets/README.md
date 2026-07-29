# Diagrams

Two kinds of SVG live here, and the distinction matters.

## Generated — never edit by hand

| File | Source of its numbers |
|---|---|
| `training-memory.svg` | `aieng.serving.budget.training_memory_gb` |
| `compounding-reliability.svg` | `aieng.nn.schedules.compounding_reliability` |
| `decode-floor.svg` | `aieng.serving.budget.decode_floor_ms` |
| `kv-cache-vs-weights.svg` | `aieng.serving.budget.kv_cache_gb` |

Every number in these charts is computed by the same functions the notes cite and the
tests assert on. A hand-drawn chart drifts from the prose within a month; a computed one
cannot.

```bash
make diagrams          # regenerate
python scripts/build_diagrams.py --check   # CI: fail if stale
```

## Hand-written — edit freely

| File | Why it is animated |
|---|---|
| `kv-cache.svg` | The work growing 1, 2, 3, 4 versus staying at 1 is the entire argument |
| `attention-causal-mask.svg` | Masking happens *before* the softmax — a sequence, not a state |
| `transformer-forward-pass.svg` | A pulse along the residual stream shows what "flows" means |
| `agent-loop.svg` | It is a loop; a static picture of a loop is a circle |

Animation is used only where motion carries information. Everything else is a Mermaid
block inline in the markdown, which is diffable and needs no build step.

## Conventions

Any SVG added here must:

- **Work in both GitHub themes.** Use the `@media (prefers-color-scheme: dark)` block and
  the shared accent palette below. A diagram that is invisible in dark mode is worse than
  no diagram.
- **Respect `prefers-reduced-motion`.** Animations resolve to their final frame rather
  than being removed, so the diagram still reads.
- **Carry `<title>` and `<desc>`.** Screen readers and search engines both use them, and
  the markdown alt text should say something different from the title, not repeat it.
- **Contain no `<script>`.** GitHub strips it, and a diagram should not need it.
- **Stay well-formed XML.** `tests/test_assets.py` parses every file here.

### Palette

Mid-saturation, chosen to hold contrast on white *and* on `#0d1117`:

| | Hex | Used for |
|---|---|---|
| blue | `#3b82f6` | inputs, embeddings, the default |
| purple | `#a855f7` | the model itself, transformer blocks |
| teal | `#14b8a6` | the good path, caching, savings |
| amber | `#f59e0b` | guards, budgets, attention-worthy |
| rose | `#f43f5e` | cost, danger, the thing that bites |
| green | `#22c55e` | output, success |

## Why not PNG

Text stays selectable and searchable, diffs are readable, files are a few kilobytes
rather than a few hundred, and rendering is sharp at any zoom. The only thing SVG costs
here is that GitHub will not run scripts inside it — which is a feature.
