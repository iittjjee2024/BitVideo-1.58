"""Tests for the TurboQuant vector-quantization codec + compressed memory store.

Verifies the properties the method relies on (arXiv:2504.19874):
  * the data-oblivious rotation is orthogonal and invertible (norm-preserving)
  * MSE reconstruction improves monotonically with bit-width
  * encode/decode round-trips preserve shape and dim
  * the inner-product estimate tracks the true inner product
  * the TurboQuant-backed store preserves retrieval recall vs fp32 keys while
    compressing the stored keys

These assert MEASURED behavior with loose, honest tolerances — not the paper's
absolute constants, which are for different settings.
"""

from __future__ import annotations

import torch

from bitmem.memory.base import MemoryItem
from bitmem.memory.turboquant import (
    RandomRotation,
    TurboCode,
    TurboQuantConfig,
    TurboQuantMSE,
    TurboQuantProd,
)
from bitmem.memory.turbo_store import TurboQuantMemoryStore
from bitmem.memory.typed import MemorySystem


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_is_invertible():
    rot = RandomRotation(48, seed=3)
    x = torch.randn(10, 48)
    recovered = rot.unrotate(rot.rotate(x))
    assert torch.allclose(x, recovered, atol=1e-4)


def test_rotation_preserves_norm():
    rot = RandomRotation(64, seed=1)
    x = torch.randn(8, 64)
    y = rot.rotate(x)
    # Orthonormal transform preserves L2 norm (padding adds only zeros).
    assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-4)


def test_rotation_pads_non_power_of_two():
    rot = RandomRotation(100, seed=0)  # 100 -> padded 128
    assert rot.padded == 128
    y = rot.rotate(torch.randn(3, 100))
    assert y.shape == (3, 128)


# ---------------------------------------------------------------------------
# MSE quantizer
# ---------------------------------------------------------------------------


def test_mse_roundtrip_shape():
    q = TurboQuantMSE(64, TurboQuantConfig(bits=4, seed=0))
    x = torch.randn(5, 64)
    recon = q.quantize(x)
    assert recon.shape == x.shape


def test_mse_reconstruction_improves_with_bits():
    X = torch.randn(128, 64)
    X = X / X.norm(dim=1, keepdim=True)
    errs = []
    for b in (2, 4, 8):
        q = TurboQuantMSE(64, TurboQuantConfig(bits=b, seed=0))
        recon = q.quantize(X)
        errs.append(float(((X - recon) ** 2).mean()))
    # Monotonic improvement: more bits -> lower reconstruction MSE.
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3  # 8-bit is near-lossless


def test_mse_bits_per_coordinate_reported():
    q = TurboQuantMSE(64, TurboQuantConfig(bits=4, seed=0))
    bpc = q.bits_per_coordinate()
    assert 4.0 <= bpc <= 6.0  # 4 payload bits + small per-vector scale overhead


# ---------------------------------------------------------------------------
# Inner-product estimator
# ---------------------------------------------------------------------------


def test_inner_product_estimate_tracks_truth():
    qp = TurboQuantProd(128, TurboQuantConfig(bits=4, qjl_bits=512, seed=0))
    A = torch.randn(64, 128)
    B = torch.randn(64, 128)
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)
    ca, cb = qp.encode(A), qp.encode(B)
    est = qp.estimate_inner_product(ca, cb)
    true = (A * B).sum(dim=1)
    # At 4 bits the estimate should be close to the true inner product.
    assert float((est - true).abs().mean()) < 0.05


# ---------------------------------------------------------------------------
# Compressed store
# ---------------------------------------------------------------------------


def _write_keys(store, keys):
    for i in range(keys.shape[0]):
        store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=1.0))


def test_turbo_store_preserves_top1_recall():
    torch.manual_seed(0)
    dim = 64
    keys = torch.randn(40, dim)
    keys = keys / keys.norm(dim=1, keepdim=True)
    store = TurboQuantMemoryStore(dim, bits=4, seed=0)
    _write_keys(store, keys)
    assert len(store) == 40
    hits = sum(1 for i in range(40) if (r := store.retrieve(keys[i], 1)) and r[0].content == i)
    # Near-perfect recall against the original fp32 keys at 4 bits.
    assert hits >= 38


def test_turbo_store_compresses():
    torch.manual_seed(0)
    dim = 128
    keys = torch.randn(50, dim)
    store = TurboQuantMemoryStore(dim, bits=4, seed=0)
    _write_keys(store, keys)
    assert store.code_bytes() < store.fp32_bytes()
    assert store.compression_ratio() > 4.0  # 4-bit keys vs fp32


def test_turbo_store_delete_prunes_codes():
    torch.manual_seed(0)
    dim = 32
    store = TurboQuantMemoryStore(dim, bits=4, seed=0)
    ids = []
    for i in range(5):
        ids.append(store.write(MemoryItem(content=i, embedding=torch.randn(dim), importance=1.0)))
    store.delete(ids[0])
    assert len(store) == 4
    assert store.code_bytes() > 0  # remaining codes accounted for


def test_memory_system_with_turboquant_backend():
    ms = MemorySystem.with_turboquant(64, bits=4, seed=0)
    assert type(ms.episodic.store).__name__ == "TurboQuantMemoryStore"
    assert type(ms.semantic.store).__name__ == "TurboQuantMemoryStore"
    torch.manual_seed(0)
    keys = torch.randn(20, 64)
    keys = keys / keys.norm(dim=1, keepdim=True)
    for i in range(20):
        ms.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=1.0),
                 memory_type="episodic")
    hits = sum(1 for i in range(20) if (r := ms.retrieve(keys[i], 1)) and r[0].content == i)
    assert hits >= 19
