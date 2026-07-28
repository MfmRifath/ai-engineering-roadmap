"""Evaluation — Huyen ch. 3-4, Lanham ch. 10.

The most under-practiced skill in the field, and the one this repo takes most
seriously. Retrieval metrics have ground truth and are cheap; generation quality
does not and is not; agent trajectories need a whole extra dimension.
"""

from aieng.evals.metrics import (
    RetrievalMetrics,
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from aieng.evals.trajectory import TrajectoryEval, check_trajectory

__all__ = [
    "RetrievalMetrics",
    "TrajectoryEval",
    "average_precision",
    "check_trajectory",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
