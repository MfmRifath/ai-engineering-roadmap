#!/usr/bin/env python3
"""Harvest flashcards out of the chapter notes into an Anki-importable deck.

Every note ends with a Flashcards section whose cards are written as:

    <!-- card -->
    Q: What does temperature control during sampling?
    A: The sharpness of the softmax over logits. T<1 concentrates probability
       mass on likely tokens; T>1 flattens it toward uniform.
    <!-- /card -->

This script walks ``books/*/notes/*.md``, pulls those blocks out, and writes a
tab-separated deck. Anki: File -> Import, choose Tab as the separator, and map
field 3 to Tags.

Usage
-----
    python scripts/build_flashcards.py
    python scripts/build_flashcards.py --book 04-ai-engineering-huyen
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"
OUT_DIR = REPO_ROOT / "flashcards"

CARD_PAT = re.compile(r"<!--\s*card\s*-->(.*?)<!--\s*/card\s*-->", re.DOTALL | re.IGNORECASE)
QA_PAT = re.compile(r"^\s*Q:\s*(.*?)\n\s*A:\s*(.*)$", re.DOTALL)


def flatten(text: str) -> str:
    """Collapse a card side to one line — Anki TSV cannot hold raw newlines."""
    text = text.strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def to_anki_html(text: str) -> str:
    """Minimal markdown -> HTML so code and emphasis survive the import."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", text)
    return text


def cards_in(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for block in CARD_PAT.findall(path.read_text(encoding="utf-8")):
        m = QA_PAT.match(block.strip())
        if not m:
            print(f"  ! malformed card in {path.name} (needs 'Q:' then 'A:')", file=sys.stderr)
            continue
        out.append((flatten(m.group(1)), flatten(m.group(2))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", help="only harvest one book directory (its slug)")
    args = ap.parse_args(argv)

    if not BOOKS_DIR.is_dir():
        print("No books/ directory yet.", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    per_book: dict[str, int] = {}

    for book_dir in sorted(BOOKS_DIR.iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        if args.book and book_dir.name != args.book:
            continue
        notes = sorted((book_dir / "notes").glob("ch*.md"))
        count = 0
        for note in notes:
            chapter = note.stem  # e.g. ch07
            tags = f"{book_dir.name} {chapter}"
            for q, a in cards_in(note):
                rows.append((to_anki_html(q), to_anki_html(a), tags))
                count += 1
        if count:
            per_book[book_dir.name] = count

    if not rows:
        print("No flashcards found. Add <!-- card --> blocks to your notes.")
        return 0

    deck = OUT_DIR / "ai-engineering-roadmap.tsv"
    with deck.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("#separator:tab\n#html:true\n#tags column:3\n")
        for q, a, tags in rows:
            fh.write(f"{q}\t{a}\t{tags}\n")

    print(f"Wrote {len(rows)} cards -> {deck.relative_to(REPO_ROOT)}")
    for book, n in per_book.items():
        print(f"  {n:>4}  {book}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
