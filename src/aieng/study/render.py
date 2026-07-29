"""Render a chapter note to HTML for reading inside the app.

Uses ``markdown-it-py`` (pure Python, no C extensions) in its GFM-like preset,
so the tables the notes rely on heavily come out as tables.

Two things need rewriting before the HTML is usable in the app:

* **Links.** A note links to its neighbours as ``../../04-.../notes/ch03.md``.
  Those become in-app routes so reading stays inside the reader instead of
  bouncing you to a raw file.
* **Images.** The diagrams are ``../../../assets/*.svg``, which the app serves
  from its own ``/assets`` mount.

Mermaid blocks render as labelled code rather than diagrams: shipping the
Mermaid bundle would mean either a CDN (breaking offline use) or vendoring
~3 MB of JavaScript. The notes' SVG diagrams are local files and do render.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from aieng.study.content import REPO_ROOT

# ../../02-hands-on-llms-alammar/notes/ch03.md  ->  02-hands-on-llms-alammar, 3
_CROSS = re.compile(r"(?:\.\./)+([0-9]{2}-[a-z0-9-]+)/notes/ch(\d{2})\.md")
_SAME = re.compile(r'href="ch(\d{2})\.md"')
_ASSET = re.compile(r'src="(?:\.\./)+assets/([^"]+)"')
_REPO_MD = re.compile(r'href="(?:\.\./)+([A-Za-z0-9_./-]+\.md)"')
_SRC_LINK = re.compile(r'href="(?:\.\./)+(src/[A-Za-z0-9_./-]+)"')


def _fallback(md_text: str) -> str:
    """A deliberately plain rendering, used when markdown-it-py is absent.

    Readable rather than pretty — better than showing the reader a stack trace
    or raw markdown.
    """
    return f"<pre class='raw-md'>{html.escape(_strip_frontmatter(md_text))}</pre>"


def render_note(md_text: str, book_slug: str) -> str:
    """Markdown -> HTML, with links and images rewritten for the app."""
    try:
        from markdown_it import MarkdownIt
    except ModuleNotFoundError:
        return _fallback(md_text)

    body = _strip_frontmatter(md_text)

    md = MarkdownIt("gfm-like", {"html": False, "linkify": False, "typographer": False})
    out = md.render(body)

    # Cross-book links -> in-app reader routes.
    out = _CROSS.sub(lambda m: f"#/read/{m.group(1)}/{int(m.group(2))}", out)
    # Same-book links, which appear as bare ch07.md.
    out = _SAME.sub(lambda m: f'href="#/read/{book_slug}/{int(m.group(1))}"', out)
    # Diagrams, served by the app.
    out = _ASSET.sub(lambda m: f'src="/assets/{m.group(1)}"', out)
    # Repo-root docs and source files -> GitHub, since the app does not render them.
    out = _REPO_MD.sub(
        lambda m: (
            f'href="https://github.com/MfmRifath/ai-engineering-roadmap/blob/main/{m.group(1)}" target="_blank" rel="noopener"'
        ),
        out,
    )
    out = _SRC_LINK.sub(
        lambda m: (
            f'href="https://github.com/MfmRifath/ai-engineering-roadmap/blob/main/{m.group(1)}" target="_blank" rel="noopener"'
        ),
        out,
    )

    # Mermaid fences arrive as <pre><code class="language-mermaid">.
    out = out.replace(
        '<pre><code class="language-mermaid">',
        '<pre class="mermaid-src"><span class="diagram-label">diagram source</span><code>',
    )
    return out


def _strip_frontmatter(text: str) -> str:
    """Drop the YAML block — the reader shows those fields in its own header."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---\n", 3)
    return text[end + 5 :] if end != -1 else text


def note_toc(md_text: str) -> list[dict]:
    """Headings, for the in-page contents sidebar.

    Skips fenced code blocks, so a ``#`` comment in a Python snippet is not
    mistaken for a heading.
    """
    toc: list[dict] = []
    in_fence = False
    for line in _strip_frontmatter(md_text).splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if m:
            title = re.sub(r"[*`\[\]]|\(.*?\)", "", m.group(2)).strip()
            toc.append({"level": len(m.group(1)), "title": title, "slug": _slug(title)})
    return toc


def _slug(title: str) -> str:
    """GitHub-style anchor slug, matching what markdown-it emits."""
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s_]+", "-", s).strip("-")


def markdown_available() -> bool:
    try:
        import markdown_it  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def assets_dir() -> Path:
    return REPO_ROOT / "assets"
