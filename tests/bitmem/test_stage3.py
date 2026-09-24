"""Stage-3 tests: typed memory, consolidation, injection Methods A/B/C.

Validates:
  - typed stores (episodic/semantic/procedural/long-term) write policies + routing
  - MemorySystem cross-type retrieval
  - consolidation: clustering, information-loss measurement, promotion, safety gates
  - Method A (cross-attention): shapes, gradients, zero-init no-op at start
  - Method C (adaptive): shapes, zero-init no-op, pooled weighting
  - MemoryAugmentedDiT: all 4 methods produce correct output shape; method is a
    config switch (ablation)

Run:
    python -m pytest tests/bitmem/test_stage3.py -v
"""

from __future__ import annotations

import time

import pytest
import torch

from bitvideo.models import VideoDiT
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem
from bitmem.memory.typed import (
    EpisodicMemory,
    LongTermMemory,
    MemorySystem,
    ProceduralMemory,
    SemanticMemory,
)
from bitmem.memory.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    cluster_spread,
    greedy_cluster,
    reconstruction_loss,
)
from bitmem.interface.adaptive import AdaptiveMemoryConditioning, pool_memory_embeddings
from bitmem.interface.cross_attn import MemoryCrossAttention
from bitmem.interface.unified import (
    InjectionMethod,
    InjectorConfig,
    MemoryAugmentedDiT,
)


def _item(dim=16, **kw) -> MemoryItem:
    emb = kw.pop("embedding", torch.randn(dim))
    return MemoryItem(content=kw.pop("content", "x"), embedding=emb, **kw)


# ---------------------------------------------------------------------------
# Typed memory
# ---------------------------------------------------------------------------


def test_semantic_rejects_low_importance():
    sem = SemanticMemory()  # importance_floor=0.3
    assert sem.write(_item(importance=0.1)) is None  # rejected
    assert sem.write(_item(importance=0.5)) is not None  # accepted


def test_episodic_accepts_everything():
    epi = EpisodicMemory()  # importance_floor=0.0
    assert epi.write(_item(importance=0.0)) is not None


def test_typed_write_tags_memory_type():
    epi = EpisodicMemory()
    item = _item(importance=0.5)
    epi.write(item)
    assert item.provenance["memory_type"] == "episodic"


def test_memory_system_routing():
    sys = MemorySystem()
    sys.write(_item(importance=0.9), memory_type="semantic")
    sys.write(_item(importance=0.1), memory_type="episodic")
    counts = sys.counts()
    assert counts["semantic"] == 1
    assert counts["episodic"] == 1
    assert sys.total() == 2


def test_memory_system_cross_type_retrieval():
    sys = MemorySystem()
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sys.write(_item(embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]), importance=0.9,
                    content="sem"), memory_type="semantic")
    sys.write(_item(embedding=torch.tensor([0.9, 0.1, 0.0, 0.0]),
                    content="epi"), memory_type="episodic")
    results = sys.retrieve(target, k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def test_reconstruction_loss_zero_for_identical():
    e = torch.tensor([1.0, 0.0, 0.0])
    cluster = [_item(embedding=e.clone()), _item(embedding=e.clone())]
    assert reconstruction_loss(e, cluster) == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_positive_for_spread():
    consolidated = torch.tensor([1.0, 0.0])
    cluster = [
        _item(embedding=torch.tensor([1.0, 0.0])),
        _item(embedding=torch.tensor([0.0, 1.0])),
    ]
    assert reconstruction_loss(consolidated, cluster) > 0.0


def test_greedy_cluster_groups_similar():
    items = [
        _item(embedding=torch.tensor([1.0, 0.0, 0.0])),
        _item(embedding=torch.tensor([0.99, 0.01, 0.0])),
        _item(embedding=torch.tensor([0.0, 0.0, 1.0])),
    ]
    clusters = greedy_cluster(items, similarity_threshold=0.9)
    # First two cluster together, third is separate.
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_consolidation_promotes_and_tracks_loss():
    epi = EpisodicMemory()
    lt = LongTermMemory()
    # Write a tight cluster of 4 near-identical memories.
    base = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for i in range(4):
        noise = torch.randn(4) * 0.01
        epi.write(_item(embedding=base + noise, importance=0.6, content=f"m{i}"))
    assert len(epi) == 4

    engine = ConsolidationEngine(
        ConsolidationConfig(similarity_threshold=0.8, min_cluster_size=3)
    )
    report = engine.consolidate(epi, into=lt)

    assert report.clusters_consolidated == 1
    assert report.memories_removed == 4
    assert report.memories_created == 1
    assert len(epi) == 0            # originals removed
    assert len(lt) == 1             # promoted to long-term
    assert 0.0 <= report.mean_information_loss < 0.1  # tight cluster => low loss


def test_consolidation_skips_loose_clusters():
    epi = EpisodicMemory()
    # Orthogonal memories should NOT be merged (high spread).
    for e in (torch.tensor([1.0, 0.0, 0.0]),
              torch.tensor([0.0, 1.0, 0.0]),
              torch.tensor([0.0, 0.0, 1.0])):
        epi.write(_item(embedding=e, importance=0.6))
    engine = ConsolidationEngine(
        ConsolidationConfig(similarity_threshold=0.1, min_cluster_size=2, max_spread=0.2)
    )
    report = engine.consolidate(epi)
    # Loose cluster exceeds max_spread -> not consolidated.
    assert report.clusters_consolidated == 0


# ---------------------------------------------------------------------------
# Method C: adaptive conditioning
# ---------------------------------------------------------------------------


def test_pool_memory_embeddings_shape():
    mems = [[_item(embedding=torch.randn(8)) for _ in range(3)], []]
    pooled = pool_memory_embeddings(mems, 8, device=torch.device("cpu"),
                                    dtype=torch.float32)
    assert pooled.shape == (2, 8)
    assert torch.allclose(pooled[1], torch.zeros(8))  # empty -> zeros


def test_adaptive_conditioning_zero_init_is_noop():
    mod = AdaptiveMemoryConditioning(memory_dim=8, conditioning_dim=16)
    t_emb = torch.randn(2, 16)
    mems = [[_item(embedding=torch.randn(8))] for _ in range(2)]
    out = mod(t_emb, mems)
    # Zero-init down projection => memory adds nothing at start.
    assert torch.allclose(out, t_emb, atol=1e-6)


def test_adaptive_conditioning_trainable_changes_output():
    mod = AdaptiveMemoryConditioning(memory_dim=8, conditioning_dim=16)
    with torch.no_grad():
        mod.down.weight.normal_(0, 0.1)  # un-zero it
    t_emb = torch.randn(2, 16)
    mems = [[_item(embedding=torch.randn(8))] for _ in range(2)]
    out = mod(t_emb, mems)
    assert not torch.allclose(out, t_emb)


# ---------------------------------------------------------------------------
# Method A: memory cross-attention
# ---------------------------------------------------------------------------


def test_memory_cross_attention_shape_and_noop():
    mod = MemoryCrossAttention(model_dim=32, memory_dim=8, num_heads=4, max_tokens=4)
    tokens = torch.randn(2, 10, 32)
    mems = [[_item(embedding=torch.randn(8))] for _ in range(2)]
    out = mod(tokens, mems)
    assert out.shape == tokens.shape
    # Zero-init gate => identity at start.
    assert torch.allclose(out, tokens, atol=1e-5)


def test_memory_cross_attention_gradients_flow():
    mod = MemoryCrossAttention(model_dim=32, memory_dim=8, num_heads=4, max_tokens=4)
    # Un-zero the gate so there is signal to backprop.
    with torch.no_grad():
        mod.attention.gate_param.fill_(0.5)
    tokens = torch.randn(2, 6, 32, requires_grad=True)
    mems = [[_item(embedding=torch.randn(8))] for _ in range(2)]
    out = mod(tokens, mems)
    out.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


# ---------------------------------------------------------------------------
# Unified injector — the ablation switch
# ---------------------------------------------------------------------------


def _tiny_dit() -> VideoDiT:
    torch.manual_seed(0)
    return VideoDiT(
        in_channels=8, dim=64, depth=2, num_heads=4,
        context_dim=64, patch_size=(1, 2, 2),
        quantization=QuantizationConfig(),
    ).eval()


@pytest.mark.parametrize("method", [
    InjectionMethod.NONE,
    InjectionMethod.MEMORY_TOKENS,
    InjectionMethod.CROSS_ATTENTION,
    InjectionMethod.ADAPTIVE,
])
def test_unified_injector_all_methods_correct_shape(method):
    dit = _tiny_dit()
    cfg = InjectorConfig(method=method, memory_dim=32, max_memory_tokens=4)
    model = MemoryAugmentedDiT(dit, cfg, quantization=QuantizationConfig()).eval()

    video = torch.randn(2, 8, 3, 16, 16)
    timesteps = torch.randint(0, 1000, (2,)).float()
    context = torch.randn(2, 8, 64)
    mems = [[_item(embedding=torch.randn(32)) for _ in range(2)] for _ in range(2)]

    with torch.no_grad():
        out = model(video, timesteps, context, mems)
    assert out.shape == video.shape
    assert torch.isfinite(out).all()


def test_unified_injector_none_ignores_memory():
    dit = _tiny_dit()
    model = MemoryAugmentedDiT(
        dit, InjectorConfig(method=InjectionMethod.NONE, memory_dim=32)
    ).eval()
    video = torch.randn(1, 8, 3, 16, 16)
    t = torch.zeros(1).float()
    context = torch.randn(1, 8, 64)
    mems = [[_item(embedding=torch.randn(32))]]
    with torch.no_grad():
        with_mem = model(video, t, context, mems)
        without_mem = model(video, t, context, None)
    # NONE method: memory is ignored, outputs identical.
    assert torch.allclose(with_mem, without_mem)


def test_unified_injector_freeze_backbone():
    dit = _tiny_dit()
    model = MemoryAugmentedDiT(
        dit, InjectorConfig(method=InjectionMethod.ADAPTIVE, memory_dim=32),
        quantization=QuantizationConfig(),
    )
    model.freeze_backbone()
    # Backbone frozen, adapter trainable.
    assert all(not p.requires_grad for p in model.dit.parameters())
    assert len(model.memory_parameters()) > 0
    assert any(p.requires_grad for p in model.memory_parameters())
