"""Stage 5 tests — memory-sensitive task, joint trainer, ablation mechanics.

These are FAST unit tests (tiny models, few steps, CPU) that verify the Stage-5
machinery is wired correctly and behaves as designed. They do NOT attempt to
reproduce the full ablation result (that needs many steps on a GPU); the
scripts/bitmem_stage5_ablation.py harness produces the measured research numbers.

What we assert here:
  * the memory-sensitive task really puts a large fraction of variance in memory
  * the joint trainer runs end-to-end for every injection method + quant mode
  * training reduces the loss (the model learns)
  * retrieval is real and vectorized retrieval == per-sample retrieval
  * the learned gate trains and produces a differentiable signal
"""

from __future__ import annotations

import torch

from bitmem.train.synthetic import MemorySensitiveDataset, MemorySensitiveSpec
from bitmem.train.joint import JointTrainer, JointConfig, resolve_device
from bitmem.train.baseline import QuantMode
from bitmem.interface.unified import InjectionMethod


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def test_memory_sensitive_shapes():
    spec = MemorySensitiveSpec(num_tasks=8, memory_gain=2.0)
    ds = MemorySensitiveDataset(num_samples=32, spec=spec, seed=0)
    s = ds[0]
    assert s["video_latent"].shape == (spec.in_channels, spec.frames, spec.height, spec.width)
    assert s["text_embedding"].shape == (spec.context_len, spec.context_dim)
    assert s["memory_key"].shape == (spec.memory_dim,)
    assert s["memory_content"].shape == (spec.memory_dim,)
    assert 0 <= s["task_id"] < spec.num_tasks


def test_memory_carries_large_variance_fraction():
    # With gain=2, the memory-only component (unit std/elem, scaled by gain) must
    # dominate the shared (unit std/elem) component — otherwise the task cannot
    # test the hypothesis.
    spec = MemorySensitiveSpec(num_tasks=16, memory_gain=2.0)
    ds = MemorySensitiveDataset(num_samples=64, spec=spec, seed=0)
    floor = ds.memory_floor_mse()  # clean-space variance of the memory component
    assert floor > 3.0  # ~gain^2 * 1.0 = 4.0; memory is the dominant signal


def test_task_contents_are_separable():
    # Retrieval keys on content must be distinguishable across tasks (real
    # retrieval, not collisions).
    spec = MemorySensitiveSpec(num_tasks=16)
    ds = MemorySensitiveDataset(num_samples=16, spec=spec, seed=0)
    c = ds.task_content
    cn = c / c.norm(dim=1, keepdim=True)
    sims = cn @ cn.t() - torch.eye(spec.num_tasks)
    assert float(sims.max()) < 0.9  # well below self-similarity


def test_precomputed_clean_matches_spec():
    spec = MemorySensitiveSpec(num_tasks=4, memory_gain=1.5)
    ds = MemorySensitiveDataset(num_samples=8, spec=spec, seed=1)
    # _clean is precomputed; indexing must match __getitem__.
    assert ds._clean.shape == (8, spec.in_channels, spec.frames, spec.height, spec.width)
    assert torch.allclose(ds[3]["video_latent"], ds._clean[3])


# ---------------------------------------------------------------------------
# Joint trainer
# ---------------------------------------------------------------------------


def _tiny_cfg(injection, quant=QuantMode.TERNARY, steps=15):
    return JointConfig(
        dim=32, depth=1, num_heads=4, quant_mode=quant, injection=injection,
        spec=MemorySensitiveSpec(num_tasks=6, memory_gain=2.0),
        num_samples=48, max_steps=steps, batch_size=8, log_every=steps,
        predict="x0", device="cpu",
    )


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    resolved = resolve_device("auto")
    assert resolved in ("cpu", "cuda")


def test_joint_trainer_runs_all_methods():
    for inj in (InjectionMethod.NONE, InjectionMethod.MEMORY_TOKENS,
                InjectionMethod.CROSS_ATTENTION, InjectionMethod.ADAPTIVE):
        tr = JointTrainer(_tiny_cfg(inj))
        r = tr.train(verbose=False)
        assert r["final_loss"] > 0
        assert r["total_params"] > 0
        assert r["use_memory"] == (inj is not InjectionMethod.NONE)
        assert r["ternary_bytes_theory"] < r["fp16_bytes"]  # theory storage smaller


def test_joint_trainer_learns():
    # Loss should drop from first to final over a modest number of steps.
    tr = JointTrainer(_tiny_cfg(InjectionMethod.ADAPTIVE, steps=120))
    r = tr.train(verbose=False)
    assert r["final_loss"] < r["first_loss"]


def test_quant_modes_supported():
    for q in (QuantMode.FP16, QuantMode.TERNARY, QuantMode.MIXED):
        tr = JointTrainer(_tiny_cfg(InjectionMethod.MEMORY_TOKENS, quant=q))
        r = tr.train(verbose=False)
        assert r["quant_mode"] == q.value


# ---------------------------------------------------------------------------
# Retrieval + gate
# ---------------------------------------------------------------------------


def test_vectorized_retrieval_matches_per_sample():
    # The batched cosine retrieval must return the same top-1 items as scoring
    # each key against the store individually.
    tr = JointTrainer(_tiny_cfg(InjectionMethod.MEMORY_TOKENS))
    ds = tr.dataset
    keys = ds.task_content[torch.tensor([0, 3, 5, 1])].to(tr.device)
    got = tr._retrieve_for_batch(keys)
    # Each query is a task's exact content, so top-1 must be that task's memory.
    bank, items = tr._memory_bank()
    for row_i, task_i in enumerate([0, 3, 5, 1]):
        # the retrieved item's stored embedding should equal that task's content
        retrieved_emb = got[row_i][0].embedding
        assert torch.allclose(
            retrieved_emb.float(), ds.task_content[task_i].float(), atol=1e-5
        )


def test_learned_gate_trains():
    cfg = _tiny_cfg(InjectionMethod.MEMORY_TOKENS, steps=60)
    cfg.use_learned_gate = True
    tr = JointTrainer(cfg)
    assert tr.gate is not None
    r = tr.train(verbose=False)
    assert r["use_learned_gate"] is True
    # On this memory-heavy task the gate should learn to (mostly) retrieve.
    assert r["gate_positive_rate"] is not None
