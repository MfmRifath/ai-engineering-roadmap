#!/usr/bin/env python3
"""Diagnose the environment and print the exact command to run next.

Written because the setup instructions assumed a working `python`, `pip`, and
`venv` — an assumption that holds on Windows and on almost nothing else.
Debian, Ubuntu, and WSL ship Python with `ensurepip` and `venv` split into
separate apt packages, so `python3 -m venv` fails with a message that is
correct but easy to miss.

Uses only the standard library, and never fails — the whole point is that it
still runs when the environment is broken.

    python3 scripts/doctor.py       # or: make doctor
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OK = "  ok   "
NO = "  --   "
WARN = "  !!   "


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    if "microsoft" in platform.release().lower():
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def debian_like() -> bool:
    return Path("/etc/debian_version").exists()


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def main() -> int:
    print()
    print("  AI Engineering Roadmap - environment check")
    print("  " + "-" * 52)

    # -- interpreter ------------------------------------------------------
    print(f"{OK}python      {sys.executable}")
    print(f"{OK}version     {platform.python_version()}")

    system = platform.system()
    if is_wsl():
        system = "WSL (Linux on Windows)"
    print(f"{OK}platform    {system}")

    if sys.version_info < (3, 10):
        print(f"{WARN}python 3.10+ is required; this is {platform.python_version()}")

    print(
        f"{OK if in_venv() else NO}virtualenv  {'active: ' + sys.prefix if in_venv() else 'not active'}"
    )

    # -- the pieces Debian splits out ------------------------------------
    has_pip = have("pip")
    has_venv = have("venv") and have("ensurepip")
    print(f"{OK if has_pip else NO}pip         {'available' if has_pip else 'MISSING'}")
    print(
        f"{OK if has_venv else NO}venv        {'available' if has_venv else 'MISSING (ensurepip absent)'}"
    )

    # -- the project ------------------------------------------------------
    has_aieng = have("aieng")
    has_fastapi = have("fastapi")
    has_torch = have("torch")
    print(f"{OK if has_aieng else NO}aieng       {'installed' if has_aieng else 'NOT installed'}")
    print(
        f"{OK if has_fastapi else NO}fastapi     {'installed (study app ready)' if has_fastapi else 'not installed'}"
    )
    print(
        f"{OK if has_torch else NO}torch       {'installed' if has_torch else 'not installed (only needed for Phase 5)'}"
    )

    make = shutil.which("make")
    print(
        f"{OK if make else NO}make        {make or 'not found - use the python commands directly'}"
    )

    library = REPO_ROOT / "library"
    pdfs = len(list(library.glob("*.pdf"))) if library.is_dir() else 0
    print(
        f"{OK if pdfs else NO}library/    {pdfs} PDF(s)"
        + ("" if pdfs else " - put your books there; `make toc` verifies them")
    )

    # -- what to do -------------------------------------------------------
    print()
    print("  next")
    print("  " + "-" * 52)

    steps: list[str] = []
    py = "python" if platform.system() == "Windows" else "python3"
    ready = has_aieng and has_fastapi

    if ready:
        # Nothing to install. Telling someone to build a venv they do not need
        # is how a diagnostic tool loses trust.
        print("  Everything the study app needs is already installed.")
        print()
        print(f"    {'make study' if make else f'{py} -m aieng.study'}")
        print()
        print("    http://127.0.0.1:8765")
        print()
        return 0

    if not has_venv or not has_pip:
        if debian_like():
            ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            pkgs = []
            if not has_venv:
                pkgs.append(f"python{ver}-venv")
            if not has_pip:
                pkgs.append("python3-pip")
            steps.append("sudo apt update")
            steps.append(f"sudo apt install -y {' '.join(pkgs)}")
            print("  Debian/Ubuntu splits venv and pip into separate packages.")
            if is_wsl():
                print("  (Running the repo from /mnt/... also works, but is slower than")
                print("   a clone inside the WSL filesystem.)")
        else:
            print(f"  This interpreter has no {'pip' if not has_pip else 'venv'}.")
            print("  Install it for your platform, or use a different interpreter.")

    if not in_venv():
        steps.append(f"{py} -m venv .venv")
        if platform.system() == "Windows":
            steps.append(".venv" + chr(92) + "Scripts" + chr(92) + "activate")
        else:
            steps.append("source .venv/bin/activate")

    if not has_aieng or not has_fastapi:
        steps.append('pip install -e ".[study]"')

    steps.append("make study" if make else f"{py} -m aieng.study")

    for step in steps:
        print(f"    {step}")

    print()
    return 0


def _quiet_probe() -> None:
    """Used by `make setup` to fail with advice rather than a raw traceback."""
    if have("pip"):
        return
    print("\n  pip is not available for this interpreter.\n", file=sys.stderr)
    if debian_like():
        print("      sudo apt install -y python3-pip python3-venv\n", file=sys.stderr)
    print("  Then run `make doctor` for the full picture.\n", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _quiet_probe()
    else:
        try:
            raise SystemExit(main())
        except Exception as exc:
            print(f"\n  doctor failed: {exc}\n", file=sys.stderr)
            print(f"  python:   {sys.executable}", file=sys.stderr)
            print(f"  version:  {sys.version}", file=sys.stderr)
            print(f"  cwd:      {os.getcwd()}", file=sys.stderr)
            raise SystemExit(1) from exc
