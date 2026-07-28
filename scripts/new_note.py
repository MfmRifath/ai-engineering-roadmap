#!/usr/bin/env python3
"""Scaffold a chapter note from templates/note.md.

Chapter titles come from the extracted PDF outlines in ``books/_toc/`` so a new
note is always titled the way the book actually titles it.

Usage
-----
    python scripts/new_note.py --book 04-ai-engineering-huyen --chapter 6
    python scripts/new_note.py --book 03-build-llm-from-scratch-raschka --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"
TOC_DIR = BOOKS_DIR / "_toc"
TEMPLATE = REPO_ROOT / "templates" / "note.md"

TOC_LINE = re.compile(r"^\s*(\d{1,2})\.\s+(.+?)\s*$")


def load_toc(slug: str) -> dict[int, str]:
    path = TOC_DIR / f"{slug}.txt"
    if not path.exists():
        print(f"No TOC for '{slug}' at {path}. Run `make toc` first.", file=sys.stderr)
        return {}
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        m = TOC_LINE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def book_title(slug: str) -> str:
    path = TOC_DIR / f"{slug}.txt"
    if path.exists():
        first = path.read_text(encoding="utf-8").splitlines()[0]
        return first.lstrip("# ").strip()
    return slug


def render(template: str, slug: str, title: str, num: int, ch_title: str) -> str:
    return (
        template.replace("{{BOOK_TITLE}}", title)
        .replace("{{BOOK_SLUG}}", slug)
        .replace("{{CH_NUM}}", str(num))
        .replace("{{CH_TITLE}}", ch_title)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--book", required=True, help="book directory slug, e.g. 04-ai-engineering-huyen"
    )
    ap.add_argument("--chapter", type=int, help="chapter number")
    ap.add_argument("--all", action="store_true", help="scaffold every missing chapter")
    ap.add_argument("--force", action="store_true", help="overwrite an existing note")
    args = ap.parse_args(argv)

    if not TEMPLATE.exists():
        print(f"Missing template at {TEMPLATE}", file=sys.stderr)
        return 2
    if not args.chapter and not args.all:
        ap.error("pass --chapter N or --all")

    toc = load_toc(args.book)
    if not toc:
        return 2

    template = TEMPLATE.read_text(encoding="utf-8")
    title = book_title(args.book)
    notes_dir = BOOKS_DIR / args.book / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(toc) if args.all else [args.chapter]
    written = skipped = 0

    for num in targets:
        if num not in toc:
            print(f"Chapter {num} is not in the TOC for {args.book}", file=sys.stderr)
            return 2
        dest = notes_dir / f"ch{num:02d}.md"
        if dest.exists() and not args.force:
            skipped += 1
            continue
        dest.write_text(render(template, args.book, title, num, toc[num]), encoding="utf-8")
        print(f"  + {dest.relative_to(REPO_ROOT)}  — {toc[num]}")
        written += 1

    print(f"{written} note(s) written, {skipped} already existed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
