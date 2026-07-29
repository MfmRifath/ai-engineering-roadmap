"""The diagrams have to survive edits — Alammar ch. 3, Huyen ch. 7 and 9.

Diagrams rot silently. A broken SVG renders as a broken-image icon on GitHub and
nothing in a normal test run notices. These checks are cheap and catch the whole
class of problem: malformed XML, a diagram that vanishes in dark mode, an
animation with no reduced-motion fallback, and charts that have drifted from the
functions that generate them.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"
SVGS = sorted(ASSETS.glob("*.svg"))

# Charts written by scripts/build_diagrams.py — these must not be hand-edited.
GENERATED = {
    "training-memory.svg",
    "compounding-reliability.svg",
    "decode-floor.svg",
    "kv-cache-vs-weights.svg",
}


def test_assets_directory_is_not_empty():
    assert SVGS, "no SVGs found — run `make diagrams`"


@pytest.mark.parametrize("svg", SVGS, ids=lambda p: p.name)
def test_svg_is_well_formed_xml(svg: Path):
    """A malformed SVG shows as a broken image and fails nothing else."""
    ET.parse(svg)


@pytest.mark.parametrize("svg", SVGS, ids=lambda p: p.name)
def test_svg_survives_dark_mode(svg: Path):
    """Every diagram must adapt to the reader's theme.

    A chart with hard-coded near-black text is invisible on GitHub's dark
    background — worse than having no diagram, because it looks like a bug.
    """
    text = svg.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in text, (
        f"{svg.name} has no dark-mode block; it will be unreadable for dark-theme readers"
    )


@pytest.mark.parametrize("svg", SVGS, ids=lambda p: p.name)
def test_animated_svgs_respect_reduced_motion(svg: Path):
    """Motion is an accessibility setting, not a preference to ignore."""
    text = svg.read_text(encoding="utf-8")
    if "@keyframes" not in text:
        pytest.skip("not animated")
    assert "prefers-reduced-motion" in text, (
        f"{svg.name} animates but has no reduced-motion fallback"
    )


@pytest.mark.parametrize("svg", SVGS, ids=lambda p: p.name)
def test_svg_is_accessible(svg: Path):
    """Title and description, so the diagram is not opaque to a screen reader."""
    root = ET.parse(svg).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert root.find(f"{ns}title") is not None, f"{svg.name} has no <title>"
    assert root.find(f"{ns}desc") is not None, f"{svg.name} has no <desc>"


@pytest.mark.parametrize("svg", SVGS, ids=lambda p: p.name)
def test_svg_contains_no_script(svg: Path):
    """GitHub strips <script> from SVGs; relying on it means a silently dead diagram."""
    assert "<script" not in svg.read_text(encoding="utf-8").lower()


def test_generated_charts_are_current():
    """The charts must match what build_diagrams.py produces right now.

    This is the check that keeps a diagram honest: if someone changes the memory
    arithmetic in ``aieng.serving.budget``, the chart asserting 56 GB of Adam
    state fails here rather than quietly contradicting the note beside it.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_diagrams.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"generated charts are stale — run `make diagrams`\n{result.stdout}\n{result.stderr}"
    )


def test_every_referenced_diagram_exists():
    """Catch an image path typo before it renders as a broken image on GitHub."""
    import re

    pattern = re.compile(r"!\[[^\]]*\]\(([^)]*assets/[^)]+\.svg)\)")
    missing: list[str] = []

    for md in REPO_ROOT.rglob("*.md"):
        if "library" in md.parts or ".git" in md.parts:
            continue
        for ref in pattern.findall(md.read_text(encoding="utf-8")):
            if not (md.parent / ref).resolve().is_file():
                missing.append(f"{md.relative_to(REPO_ROOT)} -> {ref}")

    assert not missing, "broken diagram references:\n  " + "\n  ".join(missing)


def test_generated_charts_all_present():
    names = {p.name for p in SVGS}
    assert names.issuperset(GENERATED), f"missing generated charts: {GENERATED - names}"


# ---------------------------------------------------------------------------
# Mermaid blocks
# ---------------------------------------------------------------------------

# Diagram types GitHub's Mermaid renderer understands. A typo in the first
# keyword renders as a raw code block rather than a diagram, which is easy to
# miss in review and looks broken to a reader.
MERMAID_TYPES = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "mindmap",
    "timeline",
    "quadrantChart",
    "gitGraph",
    "xychart-beta",
)


def _markdown_files() -> list[Path]:
    return [
        p
        for p in REPO_ROOT.rglob("*.md")
        if not {"library", ".git", ".venv", "node_modules"} & set(p.parts)
    ]


def _mermaid_blocks() -> list[tuple[Path, str]]:
    blocks: list[tuple[Path, str]] = []
    for md in _markdown_files():
        lines = md.read_text(encoding="utf-8").splitlines()
        buf: list[str] | None = None
        for line in lines:
            if buf is None and line.strip().startswith("```mermaid"):
                buf = []
            elif buf is not None and line.strip() == "```":
                blocks.append((md, "\n".join(buf)))
                buf = None
            elif buf is not None:
                buf.append(line)
    return blocks


def test_mermaid_blocks_exist():
    assert _mermaid_blocks(), "expected Mermaid diagrams in the docs"


def test_every_mermaid_block_declares_a_known_type():
    """A misspelled first keyword silently degrades to a plain code block."""
    bad: list[str] = []
    for md, body in _mermaid_blocks():
        first = next(
            (
                ln.strip()
                for ln in body.splitlines()
                if ln.strip() and not ln.strip().startswith("%%")
            ),
            "",
        )
        if not first.startswith(MERMAID_TYPES):
            bad.append(f"{md.relative_to(REPO_ROOT)}: {first[:60]!r}")
    assert not bad, "Mermaid blocks with an unrecognised diagram type:\n  " + "\n  ".join(bad)


def test_code_fences_are_balanced():
    """An unclosed fence swallows the rest of the document when rendered.

    Only *fence lines* count — a line that is nothing but backticks and an
    optional language tag. Backticks appearing mid-sentence (documenting a
    fence, for instance) are prose, not delimiters.
    """
    import re

    fence = re.compile(r"^\s*`{3,}[A-Za-z0-9_+-]*\s*$")
    unbalanced: list[str] = []

    for md in _markdown_files():
        lines = md.read_text(encoding="utf-8").splitlines()
        if sum(1 for ln in lines if fence.match(ln)) % 2:
            unbalanced.append(str(md.relative_to(REPO_ROOT)))

    assert not unbalanced, "odd number of code fences in:\n  " + "\n  ".join(unbalanced)
