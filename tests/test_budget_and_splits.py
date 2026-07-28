"""Memory arithmetic, schedules, splits, and collation.

These encode the numbers worth carrying in your head — Huyen ch. 7 and 9,
Geron ch. 2 and 11, Raschka ch. 7.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from aieng.classic.splits import hash_split, stratified_bins
from aieng.finetune.collate import mask_prompt_tokens, pad_batch
from aieng.nn.schedules import (
    compounding_reliability,
    cosine_with_warmup,
    linear_with_warmup,
    one_cycle,
)
from aieng.serving.budget import (
    agent_cost_usd,
    decode_floor_ms,
    inference_memory_gb,
    kv_cache_gb,
    tokens_per_second_ceiling,
    training_memory_gb,
)

# --------------------------------------------------------------------------
# Memory arithmetic — Huyen ch. 7
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precision,expected_gb",
    [("fp32", 28.0), ("fp16", 14.0), ("int8", 7.0), ("int4", 3.5)],
)
def test_inference_memory_for_7b(precision, expected_gb):
    assert inference_memory_gb(7, precision) == pytest.approx(expected_gb)


def test_adam_optimizer_state_dominates_full_finetuning():
    """The number that put full fine-tuning out of reach."""
    budget = training_memory_gb(7, precision="fp16", optimizer="adamw")
    assert budget.optimizer_gb == pytest.approx(56.0)
    assert budget.optimizer_gb > budget.weights_gb + budget.gradients_gb
    assert budget.total_gb > 80


def test_lora_removes_the_optimizer_state():
    """0.5% trainable turns 56 GB of Adam state into ~0.3 GB."""
    full = training_memory_gb(7, trainable_fraction=1.0)
    lora = training_memory_gb(7, trainable_fraction=0.005)
    assert lora.optimizer_gb < 0.5
    assert lora.total_gb < full.total_gb / 5


def test_qlora_fits_a_7b_on_a_16gb_card():
    """The result that democratized fine-tuning."""
    qlora = training_memory_gb(7, precision="nf4", trainable_fraction=0.005)
    assert qlora.fits_in(16.0)
    assert not training_memory_gb(7).fits_in(16.0)


def test_sgd_needs_no_optimizer_state():
    assert training_memory_gb(7, optimizer="sgd").optimizer_gb == 0.0


def test_budget_rejects_bad_inputs():
    with pytest.raises(ValueError):
        training_memory_gb(7, trainable_fraction=0.0)
    with pytest.raises(ValueError):
        training_memory_gb(7, precision="fp8")
    with pytest.raises(ValueError):
        training_memory_gb(7, optimizer="rmsprop")


def test_budget_string_is_readable():
    assert "TOTAL" in str(training_memory_gb(7))


# --------------------------------------------------------------------------
# KV cache and decode floor — Huyen ch. 9
# --------------------------------------------------------------------------


def test_kv_cache_for_llama_3_8b():
    # 32 layers, 8 KV heads (GQA), head_dim 128, 8k context
    assert kv_cache_gb(32, 8, 128, 8192, 1) == pytest.approx(1.074, abs=0.01)


def test_kv_cache_exceeds_the_model_at_scale():
    """At batch 32 and 8k context, the cache is bigger than fp16 weights."""
    cache = kv_cache_gb(32, 8, 128, 8192, batch_size=32)
    weights = inference_memory_gb(8, "fp16")
    assert cache > weights


def test_kv_cache_scales_linearly():
    base = kv_cache_gb(32, 8, 128, 1024, 1)
    assert kv_cache_gb(32, 8, 128, 2048, 1) == pytest.approx(2 * base)
    assert kv_cache_gb(32, 8, 128, 1024, 4) == pytest.approx(4 * base)


def test_gqa_shrinks_the_cache():
    """The entire point of grouped-query attention."""
    mha = kv_cache_gb(32, 32, 128, 8192, 1)  # every head has its own K/V
    gqa = kv_cache_gb(32, 8, 128, 8192, 1)  # 4 query heads share one K/V head
    assert gqa == pytest.approx(mha / 4)


@pytest.mark.parametrize(
    "params_b,precision,expected_ms",
    [(7, "fp16", 7.0), (7, "int4", 1.75), (70, "fp16", 70.0)],
)
def test_decode_floor(params_b, precision, expected_ms):
    """Decode must stream every weight per token — a hard bandwidth floor."""
    assert decode_floor_ms(params_b, 2.0, precision) == pytest.approx(expected_ms)


def test_quantization_halves_decode_time():
    """int8 is ~2x faster on decode purely because it is half the memory traffic."""
    fp16 = decode_floor_ms(7, 2.0, "fp16")
    int8 = decode_floor_ms(7, 2.0, "int8")
    assert int8 == pytest.approx(fp16 / 2)


def test_tokens_per_second_ceiling():
    assert tokens_per_second_ceiling(7, 2.0, "fp16") == pytest.approx(142.86, abs=0.1)


def test_agent_cost_grows_faster_than_linearly():
    """Context grows each step, so cost is not steps x single-call cost."""
    five = agent_cost_usd(2000, 500, 5, 3.0, 15.0)
    ten = agent_cost_usd(2000, 500, 10, 3.0, 15.0)
    assert ten > 2 * five


# --------------------------------------------------------------------------
# Learning-rate schedules — Geron ch. 11, Raschka Appendix D
# --------------------------------------------------------------------------


def test_warmup_ramps_up_then_peaks():
    peak = 1e-3
    assert cosine_with_warmup(0, 100, 1000, peak) < peak
    assert cosine_with_warmup(99, 100, 1000, peak) == pytest.approx(peak)


def test_cosine_decays_to_the_minimum():
    assert cosine_with_warmup(1000, 100, 1000, 1e-3) == pytest.approx(0.0, abs=1e-9)
    assert cosine_with_warmup(1000, 100, 1000, 1e-3, min_lr=1e-5) == pytest.approx(1e-5)


def test_schedule_is_monotonic_after_warmup():
    values = [cosine_with_warmup(s, 100, 1000, 1e-3) for s in range(100, 1000, 50)]
    assert all(a >= b for a, b in pairwise(values))


def test_linear_schedule_shape():
    assert linear_with_warmup(50, 100, 1000, 1e-3) < 1e-3
    assert linear_with_warmup(1000, 100, 1000, 1e-3) == pytest.approx(0.0, abs=1e-9)


def test_one_cycle_rises_then_falls():
    values = [one_cycle(s, 1000, 1e-3) for s in range(0, 1001, 50)]
    peak_index = values.index(max(values))
    assert 0 < peak_index < len(values) - 1
    assert values[0] < max(values) and values[-1] < max(values)


def test_schedules_never_exceed_the_peak():
    for step in range(0, 1000, 10):
        assert cosine_with_warmup(step, 100, 1000, 1e-3) <= 1e-3 + 1e-12
        assert one_cycle(step, 1000, 1e-3) <= 1e-3 + 1e-12


def test_compounding_reliability():
    """The arithmetic that constrains every multi-step LLM system."""
    assert compounding_reliability(0.95, 10) == pytest.approx(0.5987, abs=1e-4)
    assert compounding_reliability(0.90, 4) == pytest.approx(0.6561, abs=1e-4)
    assert compounding_reliability(1.0, 100) == 1.0
    with pytest.raises(ValueError):
        compounding_reliability(1.5, 3)


# --------------------------------------------------------------------------
# Splits — Geron ch. 2
# --------------------------------------------------------------------------


def test_hash_split_is_stable_when_the_dataset_grows():
    """The guarantee a seeded random split cannot give you."""
    small = list(range(1000))
    large = list(range(2000))

    _, test_small = hash_split(small, 0.2)
    _, test_large = hash_split(large, 0.2)

    ids_small = {small[i] for i in test_small}
    ids_large = {large[i] for i in test_large}
    # No original row may migrate between splits when rows are appended.
    assert ids_small == {i for i in ids_large if i < 1000}


def test_hash_split_ratio_is_approximately_right():
    train, test = hash_split(list(range(10_000)), 0.2)
    assert len(train) + len(test) == 10_000
    assert 0.18 < len(test) / 10_000 < 0.22


def test_hash_split_is_deterministic():
    assert hash_split(list(range(100)), 0.2) == hash_split(list(range(100)), 0.2)


def test_hash_split_works_with_string_ids():
    train, test = hash_split([f"user-{i}" for i in range(1000)], 0.2)
    assert len(train) + len(test) == 1000


def test_hash_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        hash_split([1, 2, 3], 1.5)


def test_stratified_bins():
    bins = [1.5, 3.0, 4.5, 6.0]
    assert stratified_bins([0.5, 2.0, 3.5, 5.0, 99.0], bins) == [0, 1, 2, 3, 3]


def test_stratified_bins_rejects_empty_bins():
    with pytest.raises(ValueError):
        stratified_bins([1.0], [])


# --------------------------------------------------------------------------
# Instruction-tuning collation — Raschka ch. 7
# --------------------------------------------------------------------------


def test_prompt_tokens_are_masked():
    masked = mask_prompt_tokens([5, 6, 7, 8], prompt_length=2, pad_token_id=0)
    assert masked == [-100, -100, 7, 8]


def test_first_pad_survives_as_the_stop_signal():
    """Mask every pad and the model never learns where a response ends."""
    masked = mask_prompt_tokens([7, 8, 0, 0, 0], prompt_length=0, pad_token_id=0)
    assert masked == [7, 8, 0, -100, -100]


def test_all_pads_masked_when_requested():
    masked = mask_prompt_tokens([7, 8, 0, 0], prompt_length=0, pad_token_id=0, keep_first_pad=False)
    assert masked == [7, 8, -100, -100]


def test_pad_batch_shifts_targets_by_one():
    inputs, targets = pad_batch([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert inputs[0] == [1, 2, 3]
    assert targets[0] == [2, 3, 0]  # shifted; the trailing 0 is the stop token
    assert len(inputs[0]) == len(inputs[1])  # padded to the same length


def test_pad_batch_pads_per_batch_not_to_a_global_maximum():
    short_batch, _ = pad_batch([[1, 2], [3, 4]], pad_token_id=0)
    long_batch, _ = pad_batch([[1] * 50, [2, 3]], pad_token_id=0)
    assert len(short_batch[0]) < len(long_batch[0])


def test_pad_batch_on_empty_input():
    assert pad_batch([], pad_token_id=0) == ([], [])
