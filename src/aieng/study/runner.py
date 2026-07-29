"""Run learner code in a subprocess and report which assertions passed.

**Read this before worrying about it.** This executes Python you typed, on your
machine, with your permissions — exactly like a Jupyter notebook or running
``python solution.py`` yourself. It is not a security sandbox and does not
pretend to be one. What it does provide:

* a **separate process**, so a crash or an infinite recursion cannot take the
  study app down with it;
* a **wall-clock timeout**, so an infinite loop is killed rather than hanging;
* a **temporary working directory**, so stray file writes land somewhere
  disposable rather than in your repo;
* **capped output**, so ``print`` in a loop cannot exhaust memory.

The server binds to 127.0.0.1 only, so the only person who can submit code here
is you. If that assumption ever changes, this module must not be exposed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from aieng.study.challenges import Challenge

TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 8_000
MARKER = "__AIENG_RESULTS__"


@dataclass
class TestResult:
    name: str
    passed: bool
    error: str = ""
    hidden: bool = False


@dataclass
class RunResult:
    ok: bool  # every test passed
    results: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    error: str = ""  # a failure of the run itself: syntax error, timeout
    duration_ms: int = 0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "passed": self.passed,
            "total": self.total,
            "stdout": self.stdout,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "results": [
                {"name": r.name, "passed": r.passed, "error": r.error, "hidden": r.hidden}
                for r in self.results
            ],
        }


def _indent(code: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in code.splitlines())


def build_script(user_code: str, challenge: Challenge) -> str:
    """User code first, then each test wrapped in its own function.

    Wrapping each test separately means one failure does not hide the rest —
    you see all five results, not just the first exception.
    """
    parts = [
        "import sys, json, traceback",
        "",
        "# ---- your code ----",
        user_code,
        "",
        "# ---- checks ----",
        "_results = []",
        "",
        "def _check(_name, _fn, _hidden):",
        "    try:",
        "        _fn()",
        '        _results.append({"name": _name, "passed": True, "error": "", "hidden": _hidden})',
        "    except AssertionError as _e:",
        '        _msg = str(_e) or "assertion failed"',
        '        _results.append({"name": _name, "passed": False, "error": _msg, "hidden": _hidden})',
        "    except Exception as _e:",
        '        _msg = f"{type(_e).__name__}: {_e}"',
        '        _results.append({"name": _name, "passed": False, "error": _msg, "hidden": _hidden})',
        "",
    ]

    for i, test in enumerate(challenge.tests):
        parts.append(f"def _t{i}():")
        body = _indent(test.code) or "    pass"
        parts.append(body)
        parts.append("")
        parts.append(f"_check({test.name!r}, _t{i}, {test.hidden!r})")
        parts.append("")

    parts.append(f'print("{MARKER}" + json.dumps(_results))')
    return "\n".join(parts)


def run_challenge(user_code: str, challenge: Challenge) -> RunResult:
    """Execute the learner's solution against the challenge's assertions."""
    import time

    script = build_script(user_code, challenge)
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="aieng-study-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                ok=False,
                error=(
                    f"Timed out after {TIMEOUT_SECONDS}s — most likely an infinite loop. "
                    "Check your loop's exit condition."
                ),
                duration_ms=TIMEOUT_SECONDS * 1000,
            )
        except OSError as exc:
            return RunResult(ok=False, error=f"Could not start the interpreter: {exc}")

    duration_ms = int((time.monotonic() - start) * 1000)
    stdout, marker, payload = proc.stdout.partition(MARKER)
    stdout = stdout[:MAX_OUTPUT_CHARS]

    if not marker:
        # The script died before reaching the checks — a syntax error, an
        # exception at import time, or a call to sys.exit().
        stderr = (proc.stderr or "").strip()
        return RunResult(
            ok=False,
            stdout=stdout,
            error=_clean_traceback(stderr) or "Your code exited before the checks ran.",
            duration_ms=duration_ms,
        )

    try:
        raw = json.loads(payload.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return RunResult(
            ok=False, stdout=stdout, error="Could not read the results.", duration_ms=duration_ms
        )

    results = [
        TestResult(
            name=r["name"],
            passed=r["passed"],
            error=r.get("error", ""),
            hidden=r.get("hidden", False),
        )
        for r in raw
    ]
    return RunResult(
        ok=all(r.passed for r in results) and bool(results),
        results=results,
        stdout=stdout,
        duration_ms=duration_ms,
    )


def run_scratch(user_code: str) -> RunResult:
    """Run a snippet and return its output — the scratchpad, with no checks."""
    import time

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aieng-scratch-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", user_code],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, error=f"Timed out after {TIMEOUT_SECONDS}s.")
        except OSError as exc:
            return RunResult(ok=False, error=f"Could not start the interpreter: {exc}")

    duration_ms = int((time.monotonic() - start) * 1000)
    return RunResult(
        ok=proc.returncode == 0,
        stdout=(proc.stdout or "")[:MAX_OUTPUT_CHARS],
        error=_clean_traceback((proc.stderr or "").strip()),
        duration_ms=duration_ms,
    )


def _clean_traceback(stderr: str) -> str:
    """Drop the harness frames so the error points at the learner's own code."""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    keep = [ln for ln in lines if 'File "<string>"' not in ln and "_check(" not in ln]
    return "\n".join(keep)[:MAX_OUTPUT_CHARS]
