"""Console entry points for the ``aieng`` package."""

from __future__ import annotations

import sys
from pathlib import Path


def progress_main(argv: list[str] | None = None) -> int:
    """``aieng-progress`` — rebuild PROGRESS.md from ROADMAP.md.

    Thin wrapper so the script works after ``pip install -e .`` from any
    directory, rather than only from the repo root.
    """
    repo_root = _find_repo_root()
    if repo_root is None:
        print(
            "Could not locate the roadmap repository (no ROADMAP.md found in "
            "this directory or its parents).",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(repo_root / "scripts"))
    import build_progress  # type: ignore[import-not-found]

    return build_progress.main(argv)


def _find_repo_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "ROADMAP.md").is_file():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(progress_main())
