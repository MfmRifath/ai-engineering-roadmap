# AI Engineering Roadmap — common tasks
# Windows users: run these under Git Bash, or invoke the python commands directly.

PY ?= python

.DEFAULT_GOAL := help
.PHONY: help setup setup-all test test-all lint fmt toc progress cards note clean check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install the package + dev tools (fast, no torch)
	$(PY) -m pip install -e ".[dev]"

setup-all: ## Install everything, including torch and transformers (slow)
	$(PY) -m pip install -e ".[all]"

test: ## Run the fast test suite
	$(PY) -m pytest -q -m "not slow"

test-all: ## Run every test, including slow ones
	$(PY) -m pytest -q

lint: ## Check formatting and lints
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt: ## Auto-fix lints and format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

toc: ## Re-extract chapter maps from the PDFs in library/
	$(PY) scripts/extract_toc.py --write

progress: ## Recompute PROGRESS.md and the README progress bar from ROADMAP.md
	$(PY) scripts/build_progress.py

cards: ## Rebuild the Anki-importable flashcard deck from the notes
	$(PY) scripts/build_flashcards.py

note: ## Scaffold a chapter note: make note BOOK=04-ai-engineering-huyen CH=3
	$(PY) scripts/new_note.py --book $(BOOK) --chapter $(CH)

check: lint test progress ## What CI runs

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
