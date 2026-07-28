"""Memory and latency arithmetic — Huyen ch. 7 and 9.

Two facts these functions encode, and both are worth carrying in your head:

1. **Adam's optimizer states dominate training memory.** Two fp32 moments per
   parameter is four times the weight memory in fp16. That is the number that
   puts full fine-tuning out of reach, and the reason PEFT works — eliminate
   trainable parameters and the optimizer state goes with them.

2. **Decode is memory-bandwidth-bound.** Producing one token requires reading
   every weight from memory, so there is a hard floor on time per token that no
   amount of arithmetic throughput can beat.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "nf4": 0.5,
}

# Bytes of optimizer state per *trainable* parameter.
OPTIMIZER_BYTES = {
    "adamw": 8.0,  # fp32 m and v — the villain
    "adam": 8.0,
    "sgd_momentum": 4.0,
    "sgd": 0.0,
}


@dataclass
class MemoryBudget:
    """A training memory breakdown, in gigabytes."""

    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.gradients_gb + self.optimizer_gb + self.activations_gb

    def fits_in(self, gpu_gb: float, headroom: float = 0.9) -> bool:
        """Does this fit, leaving headroom for fragmentation and workspace?"""
        return self.total_gb <= gpu_gb * headroom

    def __str__(self) -> str:
        return (
            f"weights {self.weights_gb:6.1f} GB | "
            f"grads {self.gradients_gb:6.1f} GB | "
            f"optimizer {self.optimizer_gb:6.1f} GB | "
            f"activations {self.activations_gb:6.1f} GB | "
            f"TOTAL {self.total_gb:6.1f} GB"
        )


def inference_memory_gb(n_params_b: float, precision: str = "fp16") -> float:
    """Weight memory for inference. Add the KV cache separately.

    >>> round(inference_memory_gb(7, "fp16"), 1)
    14.0
    >>> round(inference_memory_gb(7, "int4"), 1)
    3.5
    """
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"unknown precision {precision!r}")
    return n_params_b * 1e9 * BYTES_PER_PARAM[precision] / 1e9


def training_memory_gb(
    n_params_b: float,
    *,
    precision: str = "fp16",
    optimizer: str = "adamw",
    trainable_fraction: float = 1.0,
    activations_gb: float = 0.0,
) -> MemoryBudget:
    """Training memory, broken down by component.

    ``trainable_fraction`` is the PEFT knob: LoRA typically trains 0.1-1% of
    parameters, which is why gradients and optimizer state nearly vanish.

    >>> full = training_memory_gb(7)
    >>> full.total_gb > 80          # ~84 GB before activations
    True
    >>> lora = training_memory_gb(7, trainable_fraction=0.005)
    >>> lora.total_gb < 15          # ~14.4 GB
    True
    >>> qlora = training_memory_gb(7, precision="nf4", trainable_fraction=0.005)
    >>> qlora.total_gb < 5          # fits on a 16 GB card with room for activations
    True
    """
    if not 0.0 < trainable_fraction <= 1.0:
        raise ValueError("trainable_fraction must be in (0, 1]")
    if optimizer not in OPTIMIZER_BYTES:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"unknown precision {precision!r}")

    params = n_params_b * 1e9
    trainable = params * trainable_fraction
    bpp = BYTES_PER_PARAM[precision]

    return MemoryBudget(
        weights_gb=params * bpp / 1e9,
        # Gradients exist only for trainable parameters, in compute precision.
        gradients_gb=trainable * max(bpp, 2.0) / 1e9,
        optimizer_gb=trainable * OPTIMIZER_BYTES[optimizer] / 1e9,
        activations_gb=activations_gb,
    )


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: float = 2.0,
) -> float:
    """KV cache memory — usually what actually limits concurrency.

    The factor of 2 is keys **and** values. Note ``n_kv_heads``, not
    ``n_heads``: with grouped-query attention several query heads share one
    key/value head, which is precisely the point of GQA.

    Llama-3-8B (32 layers, 8 KV heads, head_dim 128) at 8k context:

    >>> round(kv_cache_gb(32, 8, 128, 8192, batch_size=1), 2)
    1.07
    >>> round(kv_cache_gb(32, 8, 128, 8192, batch_size=32), 1)
    34.4

    At batch 32 the cache exceeds the 16 GB of fp16 weights — at scale it is the
    cache, not the model, that sets your concurrency limit.
    """
    total = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size
    return total * bytes_per_element / 1e9


def decode_floor_ms(
    n_params_b: float,
    bandwidth_tb_s: float,
    precision: str = "fp16",
) -> float:
    """Lower bound on milliseconds per output token.

    Decode must stream every weight from memory for each token, so this floor is
    set by memory bandwidth and cannot be beaten by faster arithmetic. If your
    measured rate is far above this, the problem is your serving stack, not your
    hardware.

    >>> round(decode_floor_ms(7, 2.0, "fp16"), 1)      # ~143 tok/s
    7.0
    >>> round(decode_floor_ms(7, 2.0, "int4"), 2)      # ~571 tok/s
    1.75
    >>> round(decode_floor_ms(70, 2.0, "fp16"), 1)     # ~14 tok/s
    70.0
    """
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"unknown precision {precision!r}")
    bytes_read = n_params_b * 1e9 * BYTES_PER_PARAM[precision]
    return bytes_read / (bandwidth_tb_s * 1e12) * 1000


def tokens_per_second_ceiling(
    n_params_b: float, bandwidth_tb_s: float, precision: str = "fp16"
) -> float:
    """The other side of ``decode_floor_ms``, for single-stream generation."""
    return 1000.0 / decode_floor_ms(n_params_b, bandwidth_tb_s, precision)


def agent_cost_usd(
    base_tokens: int,
    tokens_per_step: int,
    steps: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Cost of an agent run — Lanham ch. 2.

    Agent cost is **not** ``steps x single_call_cost``. Each step resends a
    context grown by the previous step's observation, so input tokens accumulate
    roughly quadratically in step count. Model this before deploying.

    >>> round(agent_cost_usd(2000, 500, 10, 3.0, 15.0), 4)
    0.2025
    """
    total_input = sum(base_tokens + tokens_per_step * i for i in range(steps))
    total_output = tokens_per_step * steps
    return total_input / 1e6 * input_price_per_m + total_output / 1e6 * output_price_per_m
