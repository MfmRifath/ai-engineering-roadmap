#!/usr/bin/env python3
"""Generate the data-driven chart SVGs in ``assets/``.

Why generate rather than draw: every number in these charts comes from
``aieng.serving.budget``, the same functions the notes cite and the tests check.
A diagram that is drawn by hand drifts from the text within a month. One that is
computed cannot.

The output is deterministic, so ``--check`` can fail CI when a chart is stale.

Every chart carries its own ``@media (prefers-color-scheme: dark)`` block and
uses mid-tone accents that stay legible on both GitHub themes — a diagram that
is invisible in dark mode is worse than no diagram.

Usage
-----
    python scripts/build_diagrams.py            # write assets/*.svg
    python scripts/build_diagrams.py --check    # exit 1 if any chart is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets"

sys.path.insert(0, str(REPO_ROOT / "src"))

from aieng.nn.schedules import compounding_reliability  # noqa: E402
from aieng.serving.budget import (  # noqa: E402
    decode_floor_ms,
    inference_memory_gb,
    kv_cache_gb,
    training_memory_gb,
)

# Mid-saturation accents, chosen to hold contrast on white *and* on #0d1117.
BLUE = "#3b82f6"
PURPLE = "#a855f7"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
GREEN = "#22c55e"

STYLE = """
  <style>
    .fg   { fill: #24292f; }
    .mut  { fill: #57606a; }
    .axis { stroke: #d0d7de; stroke-width: 1; }
    .grid { stroke: #d0d7de; stroke-width: 1; stroke-dasharray: 3 3; opacity: .6; }
    .rule { stroke: #f43f5e; stroke-width: 1.5; stroke-dasharray: 5 3; }
    @media (prefers-color-scheme: dark) {
      .fg   { fill: #e6edf3; }
      .mut  { fill: #8b949e; }
      .axis { stroke: #30363d; }
      .grid { stroke: #30363d; }
    }
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .t   { font-size: 15px; font-weight: 600; }
    .st  { font-size: 11.5px; }
    .lbl { font-size: 12px; }
    .val { font-size: 12px; font-weight: 600; }
    .tick{ font-size: 10.5px; }

    .bar { transform-origin: left center; animation: grow .9s cubic-bezier(.22,1,.36,1) both; }
    @keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

    .ln { stroke-dasharray: 1200; stroke-dashoffset: 1200;
          animation: draw 1.6s ease-out forwards; }
    @keyframes draw { to { stroke-dashoffset: 0; } }

    .fade { opacity: 0; animation: fade .5s ease-out forwards; }
    @keyframes fade { to { opacity: 1; } }

    @media (prefers-reduced-motion: reduce) {
      .bar, .ln, .fade { animation: none; opacity: 1;
                         stroke-dashoffset: 0; transform: none; }
    }
  </style>
"""


def head(w: int, h: int, title: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-labelledby="t d">\n'
        f"  <title id=\"t\">{title}</title>\n"
        f"  <desc id=\"d\">{desc}</desc>\n{STYLE}"
    )


def txt(
    x: float,
    y: float,
    s: str,
    cls: str = "lbl fg",
    anchor: str = "start",
    *,
    fill: str | None = None,
    delay: float | None = None,
) -> str:
    """One text node. ``delay`` adds the fade-in animation class and its offset."""
    style = ""
    if delay is not None:
        cls = f"{cls} fade"
        style = f' style="animation-delay:{delay:.2f}s"'
    colour = f' fill="{fill}"' if fill else ""
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
        f'text-anchor="{anchor}"{colour}{style}>{s}</text>\n'
    )


# ---------------------------------------------------------------------------
# 1. Training memory: why PEFT exists
# ---------------------------------------------------------------------------


def chart_training_memory() -> str:
    full = training_memory_gb(7, precision="fp16", trainable_fraction=1.0)
    lora = training_memory_gb(7, precision="fp16", trainable_fraction=0.005)
    qlora = training_memory_gb(7, precision="nf4", trainable_fraction=0.005)

    rows = [
        ("Full fine-tune", full),
        ("LoRA  (0.5% trainable)", lora),
        ("QLoRA (nf4 + LoRA)", qlora),
    ]
    segs = [("weights_gb", BLUE, "weights"), ("gradients_gb", PURPLE, "gradients"),
            ("optimizer_gb", ROSE, "Adam states")]

    w, h = 760, 300
    x0, top, bar_h, gap = 190, 78, 34, 26
    scale = 480 / max(r[1].total_gb for r in rows)

    out = head(w, h, "Training memory for a 7B model",
               "Stacked bars comparing full fine-tuning, LoRA and QLoRA memory. "
               "Adam optimizer state dominates full fine-tuning at 56 GB.")
    out += txt(24, 30, "Training memory — 7B model", "t fg")
    out += txt(24, 50, "Adam keeps two fp32 moments per trainable parameter. That is the whole reason PEFT works.",
               "st mut")

    # 16 GB reference line
    x16 = x0 + 16 * scale
    out += f'  <line x1="{x16:.1f}" y1="{top - 10}" x2="{x16:.1f}" y2="{top + 3 * (bar_h + gap) - 12}" class="rule"/>\n'
    out += txt(x16 + 5, top - 15, "16 GB card", "tick", fill=ROSE)

    for i, (label, budget) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out += txt(x0 - 12, y + bar_h / 2 + 4, label, "lbl fg", "end")
        x = x0
        for attr, colour, _ in segs:
            val = getattr(budget, attr)
            bw = val * scale
            if bw > 0.6:
                delay = f"{0.10 * i + 0.06 * segs.index((attr, colour, _)):.2f}s"
                out += (f'  <rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                        f'fill="{colour}" rx="2" class="bar" style="animation-delay:{delay}"/>\n')
            x += bw
        out += txt(x + 8, y + bar_h / 2 + 4, f"{budget.total_gb:.0f} GB", "val fg", delay=0.9)

    ly = top + 3 * (bar_h + gap) + 4
    lx = x0
    for _, colour, name in segs:
        out += f'  <rect x="{lx}" y="{ly}" width="11" height="11" fill="{colour}" rx="2"/>\n'
        out += txt(lx + 17, ly + 10, name, "tick mut")
        lx += 26 + len(name) * 6.4
    return out + "</svg>\n"


# ---------------------------------------------------------------------------
# 2. Compounding reliability: why agents are constrained workflows
# ---------------------------------------------------------------------------


def chart_compounding() -> str:
    w, h = 760, 340
    x0, y0, pw, ph = 70, 265, 640, 195
    steps = 20

    out = head(w, h, "Compounding reliability across sequential steps",
               "Success rate falls multiplicatively with step count. At 95% per step, "
               "ten steps end to end succeed only 60% of the time.")
    out += txt(24, 30, "Compounding reliability", "t fg")
    out += txt(24, 50, "Per-step success multiplies. This is the constraint on every multi-step LLM system.", "st mut")

    for pct in range(0, 101, 25):
        y = y0 - ph * pct / 100
        out += f'  <line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" class="grid"/>\n'
        out += txt(x0 - 10, y + 4, f"{pct}%", "tick mut", "end")
    out += f'  <line x1="{x0}" y1="{y0}" x2="{x0 + pw}" y2="{y0}" class="axis"/>\n'

    for s in (1, 5, 10, 15, 20):
        x = x0 + pw * (s - 1) / (steps - 1)
        out += txt(x, y0 + 20, str(s), "tick mut", "middle")
    out += txt(x0 + pw / 2, y0 + 40, "sequential steps", "tick mut", "middle")

    series = [(0.99, GREEN, "99% per step"), (0.95, AMBER, "95% per step"), (0.90, ROSE, "90% per step")]
    for idx, (p, colour, name) in enumerate(series):
        pts = []
        for s in range(1, steps + 1):
            x = x0 + pw * (s - 1) / (steps - 1)
            y = y0 - ph * compounding_reliability(p, s)
            pts.append(f"{x:.1f},{y:.1f}")
        out += (f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
                f'stroke-width="2.5" stroke-linecap="round" class="ln" '
                f'style="animation-delay:{0.15 * idx:.2f}s"/>\n')
        end_y = y0 - ph * compounding_reliability(p, steps)
        out += txt(x0 + pw + 8, end_y + 4, name, "tick", fill=colour)

    # Call out 95% @ 10 steps = 60%
    hx = x0 + pw * 9 / (steps - 1)
    hy = y0 - ph * compounding_reliability(0.95, 10)
    out += (f'  <circle cx="{hx:.1f}" cy="{hy:.1f}" r="5" fill="{AMBER}" '
            f'class="fade" style="animation-delay:1.5s"/>\n')
    out += txt(hx + 10, hy - 10, "10 steps at 95% = 60%", "val fg", delay=1.6)
    return out + "</svg>\n"


# ---------------------------------------------------------------------------
# 3. Decode floor: LLM decoding is memory-bandwidth-bound
# ---------------------------------------------------------------------------


def chart_decode_floor() -> str:
    bw_tb = 2.0
    rows = [
        ("7B  int4", decode_floor_ms(7, bw_tb, "int4"), TEAL),
        ("7B  int8", decode_floor_ms(7, bw_tb, "int8"), GREEN),
        ("7B  fp16", decode_floor_ms(7, bw_tb, "fp16"), BLUE),
        ("70B fp16", decode_floor_ms(70, bw_tb, "fp16"), ROSE),
    ]
    w, h = 760, 320
    x0, top, bar_h, gap = 150, 92, 30, 22
    scale = 430 / max(r[1] for r in rows)

    out = head(w, h, "Decode speed floor set by memory bandwidth",
               "Milliseconds per output token for several model sizes and precisions "
               "at 2 TB/s of memory bandwidth.")
    out += txt(24, 30, "Decode floor — ms per output token", "t fg")
    out += txt(24, 50, "Every weight must be read from memory for each token generated. At 2 TB/s this is a hard floor,", "st mut")
    out += txt(24, 66, "which is why quantization speeds up decode: half the bytes, half the traffic.", "st mut")

    for i, (label, ms, colour) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out += txt(x0 - 12, y + bar_h / 2 + 4, label, "lbl fg", "end")
        out += (f'  <rect x="{x0}" y="{y}" width="{ms * scale:.1f}" height="{bar_h}" fill="{colour}" '
                f'rx="2" class="bar" style="animation-delay:{0.08 * i:.2f}s"/>\n')
        out += txt(x0 + ms * scale + 10, y + bar_h / 2 + 4,
                   f"{ms:.2f} ms   ({1000 / ms:.0f} tok/s)", "val fg", delay=0.5 + 0.08 * i)

    out += txt(x0, top + 4 * (bar_h + gap) + 14,
               "Floor = (parameters x bytes per parameter) / memory bandwidth", "tick mut")
    out += txt(x0, top + 4 * (bar_h + gap) + 32,
               "Measured far above the floor? The problem is your serving stack, not your hardware.", "tick mut")
    return out + "</svg>\n"


# ---------------------------------------------------------------------------
# 4. KV cache vs weights: what really limits concurrency
# ---------------------------------------------------------------------------


def chart_kv_cache() -> str:
    # Llama-3-8B geometry: 32 layers, 8 KV heads (GQA), head_dim 128
    cfg = dict(n_layers=32, n_kv_heads=8, head_dim=128)
    weights = inference_memory_gb(8, "fp16")
    contexts = [4096, 8192, 32768]
    batches = [1, 8, 32]

    w, h = 760, 340
    x0, y0, pw, ph = 80, 250, 560, 175
    group_w = pw / len(contexts)
    bar_w = 30
    top_val = max(kv_cache_gb(seq_len=c, batch_size=b, **cfg) for c in contexts for b in batches)
    scale = ph / top_val

    out = head(w, h, "KV cache size versus model weights",
               "KV cache memory for Llama-3-8B at several context lengths and batch sizes, "
               "compared with the 16 GB of fp16 weights.")
    out += txt(24, 30, "KV cache vs model weights — Llama-3-8B (GQA)", "t fg")
    out += txt(24, 50, "At long context and moderate batch, the cache exceeds the model. It, not the weights,", "st mut")
    out += txt(24, 66, "is what limits how many concurrent requests you can serve.", "st mut")

    wy = y0 - weights * scale
    out += f'  <line x1="{x0}" y1="{wy:.1f}" x2="{x0 + pw}" y2="{wy:.1f}" class="rule"/>\n'
    out += txt(x0 + pw + 6, wy + 4, f"weights {weights:.0f} GB", "tick", fill=ROSE)

    out += f'  <line x1="{x0}" y1="{y0}" x2="{x0 + pw}" y2="{y0}" class="axis"/>\n'
    for gb in (0, 40, 80, 120):
        y = y0 - gb * scale
        if y < 80:
            continue
        out += f'  <line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" class="grid"/>\n'
        out += txt(x0 - 10, y + 4, f"{gb}", "tick mut", "end")
    out += txt(x0 - 10, 92, "GB", "tick mut", "end")

    colours = {1: TEAL, 8: BLUE, 32: PURPLE}
    for gi, ctx in enumerate(contexts):
        gx = x0 + gi * group_w + (group_w - len(batches) * (bar_w + 6)) / 2
        for bi, batch in enumerate(batches):
            val = kv_cache_gb(seq_len=ctx, batch_size=batch, **cfg)
            bh = val * scale
            bx = gx + bi * (bar_w + 6)
            out += (f'  <rect x="{bx:.1f}" y="{y0 - bh:.1f}" width="{bar_w}" height="{bh:.1f}" '
                    f'fill="{colours[batch]}" rx="2" class="bar" '
                    f'style="transform-origin:{bx:.1f}px {y0}px; animation-delay:{0.07 * (gi * 3 + bi):.2f}s"/>\n')
            if val >= 1:
                out += txt(bx + bar_w / 2, y0 - bh - 7, f"{val:.0f}", "tick fg", "middle")
        out += txt(x0 + gi * group_w + group_w / 2, y0 + 20, f"{ctx // 1024}k context", "lbl fg", "middle")

    lx = x0
    ly = y0 + 48
    for batch in batches:
        out += f'  <rect x="{lx}" y="{ly}" width="11" height="11" fill="{colours[batch]}" rx="2"/>\n'
        out += txt(lx + 17, ly + 10, f"batch {batch}", "tick mut")
        lx += 90
    return out + "</svg>\n"


CHARTS = {
    "training-memory.svg": chart_training_memory,
    "compounding-reliability.svg": chart_compounding,
    "decode-floor.svg": chart_decode_floor,
    "kv-cache-vs-weights.svg": chart_kv_cache,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="exit 1 if any chart is stale")
    args = ap.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []

    for name, fn in CHARTS.items():
        content = fn()
        dest = ASSETS / name
        current = dest.read_text(encoding="utf-8") if dest.exists() else ""
        if current == content:
            status = "ok"
        else:
            stale.append(name)
            status = "STALE" if args.check else "written"
            if not args.check:
                dest.write_text(content, encoding="utf-8")
        print(f"  {status:>8}  assets/{name}")

    if args.check and stale:
        print(f"\n{len(stale)} chart(s) out of date — run `make diagrams`.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
