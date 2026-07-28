"""Training-loop machinery — Geron ch. 11-12, Raschka ch. 5 and Appendix D.

Learning-rate schedules are here rather than in ``transformer`` because they are
framework-independent arithmetic: warmup exists because Adam's moment estimates
are unreliable in the first few hundred steps, and applying the full learning
rate then can destabilize training permanently.
"""

from aieng.nn.schedules import cosine_with_warmup, linear_with_warmup, one_cycle

__all__ = ["cosine_with_warmup", "linear_with_warmup", "one_cycle"]
