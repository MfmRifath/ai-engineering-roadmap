"""Rank fusion and diversity — Huyen ch. 6, Alammar ch. 5 and 8.

Hybrid retrieval needs to combine a BM25 ranking with a dense ranking, and their
scores are on incomparable scales. Reciprocal Rank Fusion sidesteps that
entirely by using only **ranks**, which is why it is the standard method and why
it needs no tuning to work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists into one.

    Each document scores ``sum_i w_i / (k + rank_i)`` over the rankings it
    appears in. Because only ranks are used, BM25 scores and cosine similarities
    never have to be normalized against each other — which is exactly the
    problem that makes naive score-blending fragile.

    Parameters
    ----------
    rankings:
        Ranked lists of document ids, best first.
    k:
        Damping constant. 60 is the value from the original paper and works well
        without tuning; larger ``k`` flattens the contribution of top ranks.
    weights:
        Optional per-ranking weight, e.g. to trust dense retrieval more than BM25.

    Returns
    -------
    ``(doc_id, score)`` pairs sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def maximal_marginal_relevance(
    query_similarities: Sequence[float],
    pairwise_similarities: Sequence[Sequence[float]],
    top_k: int = 5,
    diversity: float = 0.3,
) -> list[int]:
    """Select results that are relevant **and** mutually diverse.

    Greedily pick the item maximizing::

        (1 - diversity) * sim_to_query  -  diversity * max_sim_to_already_selected

    Without this, retrieval routinely returns five near-identical chunks that
    together carry the information of one — wasting context and crowding out the
    passage that actually had the answer.

    The same objective appears in BERTopic to stop a topic label being five
    synonyms (Alammar ch. 5).

    Parameters
    ----------
    diversity:
        0.0 is pure relevance, 1.0 is pure diversity. 0.3 is a sensible default.

    Returns
    -------
    Indices of the selected items, in selection order.
    """
    n = len(query_similarities)
    if n == 0:
        return []
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be in [0, 1]")

    top_k = min(top_k, n)
    remaining = set(range(n))
    selected: list[int] = []

    first = max(remaining, key=lambda i: query_similarities[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_k and remaining:
        best_idx, best_score = None, -math.inf
        for i in remaining:
            redundancy = max(pairwise_similarities[i][j] for j in selected)
            score = (1 - diversity) * query_similarities[i] - diversity * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)  # type: ignore[arg-type]

    return selected


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, dependency-free.

    Cosine rather than Euclidean because in high dimensions distances concentrate
    — everything is roughly equidistant — while angles stay informative
    (Geron ch. 8). It also ignores magnitude, which in embeddings usually encodes
    frequency rather than meaning.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
