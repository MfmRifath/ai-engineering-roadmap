"""Learning-rate schedules — Geron ch. 11, Raschka Appendix D.

Pure functions of ``step``, so they are testable without a training run and
usable with any framework: wrap in ``torch.optim.lr_scheduler.LambdaLR``, or
just call them.

Linear warmup then cosine decay is what LLM pretraining standardized on, and
the reason is specific: Adam's second-moment estimates are noisy for the first
few hundred steps, so a full learning rate applied then can put the model in a
bad region it never recovers from.
"""

from __future__ import annotations

import math


def linear_with_warmup(
    step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float = 0.0
) -> float:
    """Ramp linearly to ``peak_lr``, then decay linearly to ``min_lr``."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + (peak_lr - min_lr) * (1.0 - progress)


def cosine_with_warmup(
    step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float = 0.0
) -> float:
    """Linear warmup, then cosine decay — the LLM training default.

    >>> round(cosine_with_warmup(0, 100, 1000, 1e-3), 6)   # start of warmup
    1e-05
    >>> round(cosine_with_warmup(99, 100, 1000, 1e-3), 6)  # end of warmup: peak
    0.001
    >>> round(cosine_with_warmup(1000, 100, 1000, 1e-3), 6)  # fully decayed
    0.0
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def one_cycle(
    step: int,
    total_steps: int,
    max_lr: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> float:
    """Smith's 1cycle — ramp up, then down, both with cosine annealing.

    Geron's recommended default for supervised training. The rise is the
    interesting part: a period at high learning rate acts as regularization and
    can dramatically shorten training.
    """
    initial_lr = max_lr / div_factor
    final_lr = initial_lr / final_div_factor
    turn = int(total_steps * pct_start)

    if step <= turn:
        progress = step / max(turn, 1)
        return initial_lr + 0.5 * (max_lr - initial_lr) * (1 - math.cos(math.pi * progress))

    progress = (step - turn) / max(total_steps - turn, 1)
    progress = min(progress, 1.0)
    return final_lr + 0.5 * (max_lr - final_lr) * (1 + math.cos(math.pi * progress))


def compounding_reliability(per_step: float, steps: int) -> float:
    """End-to-end success rate for a sequence of independent steps.

    Not a schedule, but it belongs next to them because it is the arithmetic
    that governs every multi-step LLM system (Alammar ch. 7, Huyen ch. 6,
    Lanham ch. 1). Success rates *multiply*.

    >>> round(compounding_reliability(0.95, 10), 3)
    0.599
    >>> round(compounding_reliability(0.90, 4), 3)
    0.656
    """
    if not 0.0 <= per_step <= 1.0:
        raise ValueError("per_step must be a probability")
    return per_step**steps
