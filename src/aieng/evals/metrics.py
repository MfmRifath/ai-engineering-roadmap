"""Retrieval metrics — Huyen ch. 4, Alammar ch. 8, Geron ch. 3.

Retrieval is the half of a RAG system that **has ground truth**, so it is cheap
to measure and it is where most failures live. Measuring only end-to-end answer
quality tells you the system is broken without telling you which half to fix.

These are the same precision/recall ideas from Geron ch. 3, applied to ranked
lists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Of the top k retrieved, what fraction were relevant?

    The cost of a false positive — an irrelevant chunk that consumed context and
    distracted the generator.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Of the relevant documents, what fraction appeared in the top k?

    **The metric that matters most for RAG.** Retrieval quality is the ceiling on
    system quality — the generator cannot answer from a passage it never
    received, and no prompt engineering recovers a retrieval miss.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 1.0  # vacuously satisfied; nothing was required
    return sum(1 for doc in retrieved[:k] if doc in relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result, or 0 if none.

    Right metric when the user only needs *one* good answer — a lookup, not a
    survey.
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Mean of precision@k taken at each position where a relevant doc appears.

    Rewards ranking relevant documents early, unlike recall@k which is
    indifferent to order within the cutoff.
    """
    if not relevant:
        return 1.0
    hits, total = 0, 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """Normalized discounted cumulative gain — for **graded** relevance.

    Use when documents are not simply relevant or not: a perfect answer, a
    partial answer, and a related-but-useless page deserve different credit.
    Discounts gains logarithmically by rank, then normalizes against the ideal
    ranking so scores are comparable across queries.
    """
    gains = [relevance_grades.get(doc, 0.0) for doc in retrieved[:k]]
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


@dataclass
class RetrievalMetrics:
    """The standard bundle, computed together over one query."""

    precision_at_1: float
    precision_at_5: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    map_score: float

    @classmethod
    def compute(cls, retrieved: Sequence[str], relevant: set[str]) -> RetrievalMetrics:
        return cls(
            precision_at_1=precision_at_k(retrieved, relevant, 1),
            precision_at_5=precision_at_k(retrieved, relevant, 5),
            recall_at_5=recall_at_k(retrieved, relevant, 5),
            recall_at_10=recall_at_k(retrieved, relevant, 10),
            mrr=mean_reciprocal_rank(retrieved, relevant),
            map_score=average_precision(retrieved, relevant),
        )

    @classmethod
    def average(cls, results: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
        if not results:
            raise ValueError("cannot average an empty sequence")
        n = len(results)
        return cls(
            precision_at_1=sum(r.precision_at_1 for r in results) / n,
            precision_at_5=sum(r.precision_at_5 for r in results) / n,
            recall_at_5=sum(r.recall_at_5 for r in results) / n,
            recall_at_10=sum(r.recall_at_10 for r in results) / n,
            mrr=sum(r.mrr for r in results) / n,
            map_score=sum(r.map_score for r in results) / n,
        )

    def __str__(self) -> str:
        return (
            f"P@1={self.precision_at_1:.3f}  P@5={self.precision_at_5:.3f}  "
            f"R@5={self.recall_at_5:.3f}  R@10={self.recall_at_10:.3f}  "
            f"MRR={self.mrr:.3f}  MAP={self.map_score:.3f}"
        )


def exact_match(prediction: str, reference: str, normalize: bool = True) -> bool:
    """Strictest similarity metric, and usually too strict for open-ended output."""
    if normalize:
        prediction = " ".join(prediction.lower().split())
        reference = " ".join(reference.lower().split())
    return prediction == reference


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 — a lexical similarity baseline.

    Catches paraphrase no better than it catches anything else, but it is
    deterministic, free, and a useful floor to compare a judge against.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    return 2 * precision * recall / (precision + recall)
