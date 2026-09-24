"""Stage-1 (FP16 baseline) + Stage-2 (ternary) tests for BitMem.

Validates:
  - metrics: parameter counting, theoretical-vs-fp16 storage, denoising MSE,
    efficiency report, degradation computation
  - synthetic task: determinism (same seed => same data), learnability
  - quantization mode switching (FP16/INT8/TERNARY/MIXED produce valid configs)
  - trainer convergence (FP16 baseline actually learns on the synthetic task)
  - ternary degradation is measurable (ternary final loss is a real number,
    comparable to FP16 under identical seeding)

Kept short (few steps, tiny model) so it runs on CPU in the test suite.

Run:
    python -m pytest tests/bitmem/test_stage1.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from bitmem.eval.metrics import (
    count_parameters,
    denoising_mse,
    fp16_bytes,
    measure_efficiency,
    ternary_degradation,
    theoretical_ternary_bytes,
)
from bitmem.train.baseline import (
    BaselineConfig,
    BaselineTrainer,
    QuantMode,
    quantization_for_mode,
)
from bitmem.train.synthetic import (
    SyntheticDiffusionDataset,
    SyntheticSpec,
    make_synthetic_batch,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_count_parameters():
    m = nn.Linear(10, 20)
    assert count_parameters(m) == 10 * 20 + 20  # weights + bias


def test_theoretical_ternary_vs_fp16_storage():
    n = 1_000_000
    ternary = theoretical_ternary_bytes(n)
    fp16 = fp16_bytes(n)
    # log2(3) ≈ 1.585 bits/param vs 16 bits/param -> ~10.1x smaller
    assert ternary < fp16
    ratio = fp16 / ternary
    assert 9.5 < ratio < 10.5
    # exact: 16 / log2(3)
    assert ratio == pytest.approx(16.0 / math.log2(3), rel=1e-6)


def test_degradation_positive_when_ternary_worse():
    deg = ternary_degradation(fp16_mse=0.10, ternary_mse=0.13)
    assert deg.absolute_increase == pytest.approx(0.03)
    assert deg.relative_increase == pytest.approx(0.3)


def test_degradation_negative_when_ternary_better():
    # Sometimes quantization noise acts as regularization; sign must be correct.
    deg = ternary_degradation(fp16_mse=0.10, ternary_mse=0.08)
    assert deg.absolute_increase == pytest.approx(-0.02)
    assert deg.relative_increase < 0


# ---------------------------------------------------------------------------
# Synthetic task
# ---------------------------------------------------------------------------


def test_synthetic_dataset_deterministic():
    ds1 = SyntheticDiffusionDataset(num_samples=8, seed=42)
    ds2 = SyntheticDiffusionDataset(num_samples=8, seed=42)
    for i in range(8):
        assert torch.allclose(ds1[i]["video_latent"], ds2[i]["video_latent"])
        assert torch.allclose(ds1[i]["text_embedding"], ds2[i]["text_embedding"])


def test_synthetic_dataset_different_seeds_differ():
    ds1 = SyntheticDiffusionDataset(num_samples=4, seed=1)
    ds2 = SyntheticDiffusionDataset(num_samples=4, seed=2)
    assert not torch.allclose(ds1[0]["video_latent"], ds2[0]["video_latent"])


def test_synthetic_shapes():
    spec = SyntheticSpec()
    ds = SyntheticDiffusionDataset(num_samples=2, spec=spec, seed=0)
    sample = ds[0]
    assert sample["video_latent"].shape == spec.latent_shape
    assert sample["text_embedding"].shape == (spec.context_len, spec.context_dim)


def test_synthetic_signal_is_conditioning_dependent():
    # Two different contexts must produce different clean latents (task is
    # actually conditioned on text, otherwise "learning" would be trivial).
    ds = SyntheticDiffusionDataset(num_samples=2, seed=0)
    a = ds[0]["video_latent"]
    b = ds[1]["video_latent"]
    assert not torch.allclose(a, b)


def test_make_synthetic_batch():
    batch = make_synthetic_batch(batch_size=4, seed=0)
    assert batch["video_latent"].shape[0] == 4
    assert batch["text_embedding"].shape[0] == 4


# ---------------------------------------------------------------------------
# Quantization mode switching
# ---------------------------------------------------------------------------


def test_quant_mode_fp16_disables_both():
    cfg = quantization_for_mode(QuantMode.FP16)
    assert cfg.weight.enabled is False
    assert cfg.activation.enabled is False


def test_quant_mode_ternary_enables_both():
    cfg = quantization_for_mode(QuantMode.TERNARY)
    assert cfg.weight.enabled is True
    assert cfg.activation.enabled is True
    assert cfg.weight.threshold_factor == 0.5  # packed-compatible
    assert cfg.activation.bits == 8


def test_quant_mode_int8_weights_full_precision():
    cfg = quantization_for_mode(QuantMode.INT8)
    assert cfg.weight.enabled is False
    assert cfg.activation.enabled is True


def test_quant_mode_mixed():
    cfg = quantization_for_mode(QuantMode.MIXED)
    assert cfg.weight.enabled is True
    assert cfg.activation.enabled is False


# ---------------------------------------------------------------------------
# Trainer convergence + degradation
# ---------------------------------------------------------------------------


def _tiny_config(mode: QuantMode) -> BaselineConfig:
    return BaselineConfig(
        dim=64,
        depth=2,
        num_heads=4,
        quant_mode=mode,
        num_samples=64,
        max_steps=120,
        batch_size=8,
        learning_rate=3e-4,
        log_every=1000,  # silence logging in tests
    )


def test_fp16_baseline_learns():
    """The harness must actually reduce loss (else degradation is meaningless)."""
    trainer = BaselineTrainer(_tiny_config(QuantMode.FP16))
    result = trainer.train(verbose=False)
    assert result["final_loss"] < result["first_loss"], (
        f"FP16 baseline did not learn: {result['first_loss']} -> {result['final_loss']}"
    )
    assert result["params"] > 0


def test_ternary_trains_and_degradation_measurable():
    """Ternary trains under identical seeding; degradation is a real number."""
    fp16 = BaselineTrainer(_tiny_config(QuantMode.FP16)).train(verbose=False)
    ternary = BaselineTrainer(_tiny_config(QuantMode.TERNARY)).train(verbose=False)

    # Ternary must also learn (final < first).
    assert ternary["final_loss"] < ternary["first_loss"]

    # Degradation is finite and computed.
    deg = ternary_degradation(fp16["final_loss"], ternary["final_loss"])
    assert math.isfinite(deg.relative_increase)
    assert math.isfinite(deg.absolute_increase)


def test_efficiency_report_fields():
    trainer = BaselineTrainer(_tiny_config(QuantMode.TERNARY))
    batch = make_synthetic_batch(batch_size=4, seed=0)
    eff = measure_efficiency(
        trainer.model, batch, label="ternary", device=torch.device("cpu")
    )
    assert eff.total_params > 0
    assert eff.theoretical_ternary_mb < eff.fp16_equivalent_mb
    assert eff.latency_ms > 0
    assert eff.samples_per_sec > 0
    row = eff.as_row()
    assert row["label"] == "ternary"


def test_denoising_mse_runs():
    trainer = BaselineTrainer(_tiny_config(QuantMode.FP16))
    batch = make_synthetic_batch(batch_size=4, seed=0)
    torch.manual_seed(0)
    mse = denoising_mse(trainer.model, batch, device=torch.device("cpu"))
    assert math.isfinite(mse)
    assert mse >= 0
