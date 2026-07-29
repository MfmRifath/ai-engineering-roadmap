# Contributing

This is primarily a personal learning log, but corrections and better explanations
are welcome.

## The one rule

**Never commit book content.**

No PDFs, no EPUBs, no extracted text, no transcribed passages, no scanned figures.
The notes here are original summaries — they paraphrase and connect ideas, they do
not reproduce the books.

CI enforces the file-extension case (`.pdf`, `.epub`, …) and fails if the repo pack
exceeds 50 MB, but **it cannot detect transcribed prose**. That part is on you. If
a paragraph reads like it was copied, it does not belong here.

Your own copies live in `library/`, which is gitignored.

## What is most useful

1. **Corrections.** If a note is wrong, that matters more than anything else here.
   Open an issue or a PR with the fix and a reason.
2. **Better explanations.** If you can explain something more clearly in fewer
   words, that is a real improvement.
3. **Dated material.** The field moves. If a note describes an API or a practice
   that has since changed, flag it — mark the old version `[dated]` and note the
   modern equivalent rather than deleting the history.
4. **Broken cross-links.** The notes reference each other heavily.
5. **Tests.** A test that catches a real mistake is worth more than a new feature.

## Note conventions

Every chapter note follows the same shape ([`templates/note.md`](templates/note.md)),
because that is what makes the flashcard and progress tooling mechanical:

- YAML frontmatter with `book`, `chapter`, `status`, `difficulty`, `est_hours`
- **Why this chapter matters** → **Core concepts** → **Key code** → **Gotchas** →
  **Exercises** → **Flashcards** → **Cross-links**

Scaffold a new one rather than copying by hand:

```bash
make note BOOK=04-ai-engineering-huyen CH=6
```

**Flashcards** go in `<!-- card -->` fences with `Q:` then `A:`. `make cards`
harvests them; malformed blocks are reported.

**`[dated]`** marks anything that has aged — an API, a benchmark number, a specific
model. Say what replaced it.

**`[framework-specific]`** marks material tied to a library that will churn.

## Diagram conventions

Flowcharts go inline as ```` ```mermaid ```` blocks — they render natively on GitHub, diff
cleanly, and need no build step. Reach for an SVG in [`assets/`](assets/) only when motion
or precise layout carries information a flowchart cannot.

The four charts in `assets/` ending in memory, reliability, floor, and weights are
**generated** from `aieng.serving.budget` — edit the script, not the SVG:

```bash
make diagrams
```

Anything hand-written in `assets/` must work in both GitHub themes, respect
`prefers-reduced-motion`, and carry `<title>` and `<desc>`. See
[assets/README.md](assets/README.md); `tests/test_assets.py` enforces all of it.

## Code conventions

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup      # install + dev tools
make check      # what CI runs: lint, format, tests, progress freshness
```

`make` resolves `python3` before `python`, and prefers an activated virtualenv —
macOS and most Linux distributions have no bare `python`. `make python` shows what
it picked; `make setup PY=/path/to/python` overrides it.

- Ruff for lint and format, 100 columns. **Markdown is excluded** — the Python
  snippets in the notes are hand-formatted for reading, and reflowing them makes
  them worse.
- Heavy dependencies (torch, transformers) are **optional extras**, imported lazily.
  `import aieng` must stay fast and work without them.
- Comments explain *why*, not *what*. The notes explain the what.
- Tests should assert a property that would catch a real mistake, not just exercise
  a line.

## Before opening a PR

```bash
make check
git ls-files | grep -Ei '\.(pdf|epub|mobi)$'   # must return nothing
```

If you ticked boxes in `ROADMAP.md`, run `make progress` so `PROGRESS.md` and the
README bar stay in sync — CI checks this.

## Licensing

By contributing you agree that code is [MIT](LICENSE) and prose is
[CC BY-NC-SA 4.0](LICENSE-CONTENT).
