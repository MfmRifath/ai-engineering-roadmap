#!/usr/bin/env python3
"""Extract chapter outlines from the PDFs in ``library/``.

Why this exists
---------------
The chapter maps in ``books/*/README.md`` claim to mirror each book's real
structure. This script is the receipt: it reads the PDF outline (bookmark)
tree straight out of the files and prints what is actually there, so a note
that drifts out of sync with its source can be caught mechanically.

It has **no dependencies** — it parses the PDF object graph directly, walking
uncompressed regions and inflating FlateDecode streams to find ``/Title``
entries. That keeps ``make toc`` runnable on a bare Python install.

Usage
-----
    python scripts/extract_toc.py             # print outlines to stdout
    python scripts/extract_toc.py --verify    # exit 1 if chapter counts drift
    python scripts/extract_toc.py --write     # refresh books/_toc/*.txt
    python scripts/extract_toc.py --full      # every outline entry, not just chapters
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = REPO_ROOT / "library"
TOC_DIR = REPO_ROOT / "books" / "_toc"

# A PDF string follows /Title either as (literal) or <hex>.
TITLE_PAT = re.compile(rb"/Title\s*[(<]")
STREAM_PAT = re.compile(rb"stream\r?\n")

MAX_TITLE_BYTES = 500
MAX_STREAM_BYTES = 6_000_000

# Each book keyed by a substring of its filename, with the chapter-heading
# pattern that book's outline actually uses and the expected chapter count.
BOOKS: dict[str, dict[str, object]] = {
    "geron": {
        "match": "Aur",
        "slug": "01-hands-on-ml-geron",
        "title": "Hands-On Machine Learning (2nd ed)",
        "pattern": re.compile(r"^Chapter (\d{1,2}): (.+)$"),
        "expected": 19,
    },
    "alammar": {
        "match": "Alammar",
        "slug": "02-hands-on-llms-alammar",
        "title": "Hands-On Large Language Models",
        "pattern": re.compile(r"^Chapter (\d{1,2})\. (.+)$"),
        "expected": 12,
    },
    "raschka": {
        "match": "Raschka",
        "slug": "03-build-llm-from-scratch-raschka",
        "title": "Build a Large Language Model (From Scratch)",
        "pattern": re.compile(r"^(\d{1,2}) ([A-Z].+)$"),
        "expected": 7,
    },
    "huyen": {
        "match": "Chip Huyen",
        "slug": "04-ai-engineering-huyen",
        "title": "AI Engineering",
        "pattern": re.compile(r"^Chapter (\d{1,2})\. (.+)$"),
        "expected": 10,
    },
    "lanham": {
        "match": "Lanham",
        "slug": "05-ai-agents-in-action-lanham",
        "title": "AI Agents in Action",
        "pattern": re.compile(r"^(\d{1,2}) ([A-Z].+)$"),
        "expected": 11,
    },
}


def _decode(raw: bytes) -> str:
    """Decode a PDF text string. UTF-16BE is flagged by a BOM; else PDFDocEncoding."""
    if raw[:2] == b"\xfe\xff":
        return raw[2:].decode("utf-16-be", "replace")
    return raw.decode("latin-1", "replace")


def _read_literal(buf: bytes, start: int) -> bytes:
    """Read a balanced ( ... ) literal string, honouring backslash escapes."""
    depth = 1
    i = start + 1
    out: list[bytes] = []
    while i < len(buf) and i - start < MAX_TITLE_BYTES:
        ch = buf[i : i + 1]
        if ch == b"\\":
            out.append(buf[i + 1 : i + 2])
            i += 2
            continue
        if ch == b"(":
            depth += 1
        elif ch == b")":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return b"".join(out)


def _titles_in(buf: bytes) -> list[str]:
    """Pull every /Title value out of a byte range."""
    found: list[str] = []
    for match in TITLE_PAT.finditer(buf):
        i = match.end() - 1
        if buf[i : i + 1] == b"<":
            close = buf.find(b">", i)
            if close < 0 or close - i > MAX_TITLE_BYTES:
                continue
            try:
                found.append(_decode(bytes.fromhex(buf[i + 1 : close].decode("ascii"))))
            except ValueError:
                continue
        else:
            found.append(_decode(_read_literal(buf, i)))
    return found


def outline_entries(pdf: Path) -> list[str]:
    """Return the PDF's outline titles, in document order, de-duplicated."""
    data = pdf.read_bytes()
    titles = _titles_in(data)

    # Outlines in modern PDFs usually live in compressed object streams.
    for match in STREAM_PAT.finditer(data):
        start = match.end()
        end = data.find(b"endstream", start)
        if end < 0 or end - start > MAX_STREAM_BYTES:
            continue
        try:
            titles.extend(_titles_in(zlib.decompress(data[start:end])))
        except zlib.error:
            continue

    seen: set[str] = set()
    ordered: list[str] = []
    for title in titles:
        norm = " ".join(title.split())
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered


def find_pdf(needle: str) -> Path | None:
    if not LIBRARY.is_dir():
        return None
    for pdf in sorted(LIBRARY.glob("*.pdf")):
        if needle.lower() in pdf.name.lower():
            return pdf
    return None


def chapters(entries: list[str], pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """Keep only top-level chapter headings, first occurrence wins."""
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for entry in entries:
        m = pattern.match(entry)
        if not m:
            continue
        num = int(m.group(1))
        title = m.group(2).strip()
        # Sub-sections like "2.1 Mastering the API" carry a dot; chapters do not.
        if "." in entry.split(" ")[0].rstrip("."):
            continue
        if num in seen or len(title) > 90:
            continue
        seen.add(num)
        out.append((num, title))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--verify", action="store_true", help="exit non-zero if chapter counts drift")
    ap.add_argument("--write", action="store_true", help="write books/_toc/<slug>.txt")
    ap.add_argument("--full", action="store_true", help="print every outline entry")
    args = ap.parse_args(argv)

    if not LIBRARY.is_dir():
        print(f"No library/ directory at {LIBRARY}.", file=sys.stderr)
        print("Put your PDFs there (it is gitignored). See books/MANIFEST.md.", file=sys.stderr)
        return 2

    problems = 0
    for key, spec in BOOKS.items():
        pdf = find_pdf(str(spec["match"]))
        print("=" * 78)
        print(f"{spec['title']}  [{key}]")
        if pdf is None:
            print("  !! PDF not found in library/ — skipping")
            problems += 1
            continue

        entries = outline_entries(pdf)
        found = chapters(entries, spec["pattern"])  # type: ignore[arg-type]
        expected = int(spec["expected"])  # type: ignore[call-overload]
        flag = "OK " if len(found) == expected else "!! "
        print(f"  {flag}{len(found)} chapters found (expected {expected})")
        if len(found) != expected:
            problems += 1

        for num, title in found:
            print(f"    {num:>2}. {title}")

        if args.full:
            print("  --- full outline ---")
            for entry in entries:
                print(f"      {entry}")

        if args.write:
            TOC_DIR.mkdir(parents=True, exist_ok=True)
            dest = TOC_DIR / f"{spec['slug']}.txt"
            body = "\n".join(f"{n:>2}. {t}" for n, t in found)
            dest.write_text(
                f"# {spec['title']}\n"
                f"# Extracted from the PDF outline by scripts/extract_toc.py\n"
                f"# Do not edit by hand — run `make toc`.\n\n{body}\n",
                encoding="utf-8",
            )
            print(f"  -> wrote {dest.relative_to(REPO_ROOT)}")

    if args.verify and problems:
        print(f"\nFAILED: {problems} book(s) drifted from expectations.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
