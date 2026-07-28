"""Chunking, fusion, and retrieval metrics — Alammar ch. 8, Huyen ch. 4 and 6."""

from __future__ import annotations

from itertools import pairwise

import pytest

from aieng.evals.metrics import (
    RetrievalMetrics,
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)
from aieng.rag.chunking import chunk_by_structure, chunk_text, estimate_tokens
from aieng.rag.fusion import (
    cosine_similarity,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)

TEXT = " ".join(f"word{i}" for i in range(400))


# --------------------------------------------------------------------------
# Chunking invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size,overlap", [(100, 0), (100, 20), (256, 64), (512, 51)])
def test_every_chunk_respects_the_size_bound(size, overlap):
    for chunk in chunk_text(TEXT, chunk_size=size, overlap=overlap):
        assert len(chunk.text) <= size


def test_chunks_cover_the_whole_document():
    """Nothing may be silently dropped."""
    chunks = chunk_text(TEXT, chunk_size=200, overlap=40)
    recovered = " ".join(c.text for c in chunks)
    for word in ("word0", "word200", "word399"):
        assert word in recovered


def test_overlap_actually_overlaps():
    """Without overlap an answer spanning a boundary becomes unretrievable."""
    chunks = chunk_text(TEXT, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    shared = 0
    for a, b in pairwise(chunks):
        if set(a.text.split()) & set(b.text.split()):
            shared += 1
    assert shared >= len(chunks) - 2


def test_zero_overlap_produces_disjoint_chunks():
    chunks = chunk_text(TEXT, chunk_size=200, overlap=0)
    for a, b in pairwise(chunks):
        assert not (set(a.text.split()) & set(b.text.split()))


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("short", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_empty_text_produces_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n  ", 100, 10) == []


def test_chunk_indices_are_sequential():
    chunks = chunk_text(TEXT, chunk_size=150, overlap=30)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_rejects_overlap_at_or_above_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(TEXT, chunk_size=100, overlap=100)


def test_structural_chunking_keeps_paragraphs_intact():
    doc = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    chunks = chunk_by_structure(doc, max_chunk_size=200)
    assert len(chunks) == 1  # all three fit in one chunk
    assert "First" in chunks[0].text and "Third" in chunks[0].text


def test_structural_chunking_falls_back_on_oversized_paragraphs():
    doc = "Short one.\n\n" + ("long " * 300)
    chunks = chunk_by_structure(doc, max_chunk_size=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_contextual_retrieval_prefix():
    chunk = chunk_text("some content here", 100, 0)[0]
    assert chunk.with_context("A paper about X").startswith("[Document context:")


def test_token_estimate_is_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# --------------------------------------------------------------------------
# Reciprocal rank fusion
# --------------------------------------------------------------------------


def test_rrf_rewards_documents_ranked_well_by_both_retrievers():
    bm25 = ["a", "b", "c", "d"]
    dense = ["c", "a", "e", "f"]
    fused = reciprocal_rank_fusion([bm25, dense])
    assert fused[0][0] == "a"  # 1st and 2nd beats 3rd and 1st


def test_rrf_includes_documents_found_by_only_one_retriever():
    """The reason hybrid works: BM25 finds exact IDs dense retrieval misses."""
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "d"]]))
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_needs_no_score_normalization():
    """Only ranks are used, so incomparable score scales never matter."""
    ranking = ["x", "y", "z"]
    assert reciprocal_rank_fusion([ranking]) == reciprocal_rank_fusion([ranking])


def test_rrf_weights_shift_the_ordering():
    bm25, dense = ["a", "b"], ["b", "a"]
    trust_dense = reciprocal_rank_fusion([bm25, dense], weights=[0.2, 1.0])
    assert trust_dense[0][0] == "b"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_on_empty_input():
    assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------------------
# MMR
# --------------------------------------------------------------------------


def test_mmr_picks_the_most_relevant_item_first():
    sims = [0.9, 0.8, 0.7]
    pair = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert maximal_marginal_relevance(sims, pair, top_k=2)[0] == 0


def test_mmr_avoids_near_duplicates():
    """Items 0 and 1 are nearly identical; a diverse selection should skip one."""
    sims = [0.90, 0.89, 0.60]
    pair = [
        [1.00, 0.99, 0.10],
        [0.99, 1.00, 0.10],
        [0.10, 0.10, 1.00],
    ]
    selected = maximal_marginal_relevance(sims, pair, top_k=2, diversity=0.7)
    assert selected == [0, 2]


def test_mmr_with_zero_diversity_is_pure_relevance():
    sims = [0.90, 0.89, 0.60]
    pair = [[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]]
    assert maximal_marginal_relevance(sims, pair, top_k=2, diversity=0.0) == [0, 1]


def test_mmr_handles_top_k_larger_than_input():
    sims = [0.5, 0.4]
    pair = [[1.0, 0.2], [0.2, 1.0]]
    assert len(maximal_marginal_relevance(sims, pair, top_k=10)) == 2


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # no division by zero


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------


def test_precision_and_recall_at_k():
    retrieved = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c", "d"}

    assert precision_at_k(retrieved, relevant, 1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.6)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(0.75)  # 3 of 4
    assert recall_at_k(retrieved, relevant, 1) == pytest.approx(0.25)


def test_recall_is_indifferent_to_order_within_k():
    relevant = {"a", "b"}
    assert recall_at_k(["a", "b", "x"], relevant, 3) == recall_at_k(["x", "b", "a"], relevant, 3)


def test_mrr_uses_the_first_relevant_rank():
    assert mean_reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["a", "x"], {"a"}) == pytest.approx(1.0)
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_average_precision_rewards_early_relevant_results():
    early = average_precision(["a", "b", "x", "y"], {"a", "b"})
    late = average_precision(["x", "y", "a", "b"], {"a", "b"})
    assert early > late


def test_ndcg_handles_graded_relevance():
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    perfect = ndcg_at_k(["a", "b", "c"], grades, 3)
    reversed_order = ndcg_at_k(["c", "b", "a"], grades, 3)
    assert perfect == pytest.approx(1.0)
    assert reversed_order < perfect


def test_metrics_bundle_and_average():
    m1 = RetrievalMetrics.compute(["a", "b"], {"a"})
    m2 = RetrievalMetrics.compute(["x", "a"], {"a"})
    avg = RetrievalMetrics.average([m1, m2])
    assert 0.0 <= avg.mrr <= 1.0
    assert "MRR" in str(avg)


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)


def test_token_f1():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert token_f1("the cat", "a dog") == 0.0
    assert 0.0 < token_f1("the cat sat", "the cat stood") < 1.0
