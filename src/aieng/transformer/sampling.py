"""Generation and sampling — Raschka ch. 5, Alammar ch. 6, Huyen ch. 2.

Temperature and top-k are implemented here rather than described, because the
mechanics are the point: temperature divides the **logits** before the softmax,
and top-k sets rejected logits to ``-inf`` before the softmax — the same trick
as causal masking, which is worth noticing.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_simple(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy decoding. Deterministic, and repetitive — that is the point of ch. 5.

    Note the two details that matter:

    * the context is **cropped** to ``context_size``, because positions beyond it
      have no positional embedding;
    * only the **last** position's logits are used. Every position is computed,
      but only the final one predicts the next token.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generation with temperature, top-k, and top-p.

    Parameters
    ----------
    temperature:
        Divides the logits before the softmax. ``< 1`` sharpens toward the argmax
        (conservative); ``> 1`` flattens toward uniform (diverse). ``0.0`` means
        greedy. It changes randomness, never competence — raising temperature
        will not fix a wrong answer.
    top_k:
        Keep only the ``k`` highest-probability tokens. A hard cap that ignores
        the shape of the distribution.
    top_p:
        Nucleus sampling — keep the smallest set whose cumulative probability
        reaches ``p``. Generally preferable to top-k because it **adapts**: when
        the model is confident it admits few candidates, when uncertain it
        admits many.
    eos_id:
        Stop early if this token is produced.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_keep = top_logits[:, -1].unsqueeze(-1)
            # -inf before softmax, exactly as with the causal mask
            logits = torch.where(logits < min_keep, torch.tensor(-torch.inf), logits)

        if top_p is not None:
            logits = _apply_top_p(logits, top_p)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: mask everything outside the smallest set summing to top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Keep tokens up to and including the one that crosses the threshold, so at
    # least one token always survives even if the top token exceeds top_p.
    remove_sorted = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, -torch.inf)


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """Encode to a (1, T) batch. Works with ``tiktoken`` or ``aieng.tokenizer``."""
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    """Cross-entropy over every position — the pretraining loss.

    ``cross_entropy`` expects **raw logits**, not probabilities: it applies
    log-softmax internally with the numerically stable log-sum-exp trick.
    Passing softmax output applies the operation twice, trains badly, and
    raises nothing.
    """
    import torch.nn.functional as F

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def perplexity(loss: float | torch.Tensor) -> float:
    """exp(cross-entropy) — the effective number of tokens being chosen among.

    An untrained model over a 50,257-token vocabulary has loss ~10.8
    (``ln(50257)``) and perplexity ~50,000: it is guessing uniformly. Loss 3.0
    means perplexity ~20 — about twenty plausible options.
    """
    value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
    return float(torch.exp(torch.tensor(value)))
