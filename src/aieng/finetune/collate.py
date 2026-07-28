"""Instruction-tuning collation — Raschka ch. 7.

The two details that decide whether an SFT run works:

* **Loss masking.** ``-100`` is ``cross_entropy``'s default ``ignore_index``, so
  those positions contribute no loss and no gradient. Without it the model
  spends capacity learning to emit padding, and learning to reproduce
  instructions rather than answer them.

* **Keeping the first pad token unmasked.** That token is the end-of-sequence
  signal. Mask every pad and the model never learns where a response *stops*,
  so it generates until it hits the token limit.
"""

from __future__ import annotations

IGNORE_INDEX = -100


def mask_prompt_tokens(
    target_ids: list[int],
    prompt_length: int,
    pad_token_id: int,
    *,
    ignore_index: int = IGNORE_INDEX,
    keep_first_pad: bool = True,
) -> list[int]:
    """Mask prompt tokens and padding in a target sequence.

    Parameters
    ----------
    prompt_length:
        Number of leading tokens belonging to the instruction. Pass ``0`` to
        train on the full sequence — the research is genuinely mixed on which is
        better, though completion-only is the common default.
    keep_first_pad:
        Leave the first padding token as a real target so the model learns to
        terminate.

    >>> mask_prompt_tokens([5, 6, 7, 8, 0, 0, 0], prompt_length=2, pad_token_id=0)
    [-100, -100, 7, 8, 0, -100, -100]
    """
    out = list(target_ids)

    for i in range(min(prompt_length, len(out))):
        out[i] = ignore_index

    seen_pad = False
    for i, token in enumerate(target_ids):
        if token != pad_token_id or i < prompt_length:
            continue
        if not seen_pad and keep_first_pad:
            seen_pad = True  # this one stays: it is the stop signal
            continue
        out[i] = ignore_index

    return out


def pad_batch(
    sequences: list[list[int]],
    pad_token_id: int,
    *,
    max_length: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pad to the longest sequence **in this batch**, and build shifted targets.

    Per-batch padding rather than padding to the dataset maximum avoids
    computing over pad tokens no example in the batch needs — a substantial
    saving when lengths vary.

    Returns ``(inputs, targets)`` where ``targets`` is ``inputs`` shifted by one.
    """
    if not sequences:
        return [], []

    batch_max = max(len(s) for s in sequences) + 1
    if max_length is not None:
        batch_max = min(batch_max, max_length + 1)

    inputs, targets = [], []
    for seq in sequences:
        padded = [*seq, pad_token_id][:batch_max]
        padded = padded + [pad_token_id] * (batch_max - len(padded))
        inputs.append(padded[:-1])
        targets.append(padded[1:])  # the off-by-one that matters
    return inputs, targets
