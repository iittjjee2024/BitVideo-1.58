"""TurboQuant — data-oblivious vector quantization for the BitMem store.

Faithful implementation of the core method from

    TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / DeepMind),
    arXiv:2504.19874, ICLR 2026.

The idea (paraphrased for licensing compliance — see the paper for the full
treatment and the formal distortion bounds):

  1. Randomly ROTATE each input vector. In high dimensions the rotated
     coordinates become near-independent and concentrate onto a predictable
     (Beta-like) distribution, so no per-vector calibration is needed. The
     rotation is DATA-OBLIVIOUS: one fixed random transform works for every
     vector, which is what makes the method "online" (no training/codebook).

  2. Quantize each rotated coordinate with a simple OPTIMAL SCALAR QUANTIZER at
     the target bit-width. Because the coordinates are concentrated and
     near-i.i.d., per-coordinate scalar quantization is already near-optimal for
     mean-squared error. This is `TurboQuantMSE`.

  3. MSE-optimal quantization is BIASED for inner-product estimation. To get an
     UNBIASED inner-product estimate (what cosine/dot-product retrieval needs),
     add a 1-bit Quantized-Johnson-Lindenstrauss (QJL) transform on the residual
     error. This is `TurboQuantProd`.

Why this belongs in BitMem: the memory store keeps high-dimensional retrieval
keys and scores them by inner product / cosine. TurboQuant compresses those keys
toward the information-theoretic limit while preserving the geometry retrieval
depends on — exactly the paper's "vector database / nearest-neighbor search" use
case. It is a VECTOR quantizer for stored embeddings; it is NOT the ternary
WEIGHT quantizer used in the DiT (a different problem).

Honesty notes (prototype, §16):
  * We use a randomized Hadamard transform (SRHT: random sign flip + Hadamard +
    subsample/pad) as the data-oblivious rotation. It is O(d log d) and gives the
    coordinate-mixing the method relies on. The paper's guarantees are stated for
    a Haar-random rotation; the Hadamard variant is the standard practical stand-in.
  * We implement a uniform scalar quantizer with a per-vector scale (derived from
    the rotated-vector norm), which is the simple near-optimal choice the method
    prescribes. We do NOT reproduce the paper's exact companding tables.
  * Bits-per-coordinate and reconstruction error are MEASURED by the benchmark,
    not asserted. We do not claim the paper's absolute numbers for our setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Data-oblivious random rotation (SRHT)
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _hadamard(n: int, device, dtype) -> torch.Tensor:
    """Dense (unnormalized) Hadamard matrix of size n (n a power of two)."""
    assert n & (n - 1) == 0, "n must be a power of two"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


class RandomRotation:
    """Data-oblivious orthogonal rotation via a randomized Hadamard transform.

    Transform: pad to the next power of two D', apply a random {+-1} diagonal sign
    flip, then the normalized Hadamard matrix. The result is an orthonormal map
    on R^{D'}; we embed R^{d} into R^{D'} by zero-padding. One fixed instance is
    reused for every vector (hence "data-oblivious" / online).

    Stored state is just the seed + dims, so the transform is cheap to persist
    and reproduce.
    """

    def __init__(self, dim: int, *, seed: int = 0, device=None, dtype=torch.float32) -> None:
        self.dim = int(dim)
        self.padded = _next_pow2(self.dim)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        g = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (self.padded,), generator=g).to(dtype) * 2 - 1
        self._signs = signs.to(device=device)
        self._H = _hadamard(self.padded, device, dtype) / math.sqrt(self.padded)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.padded:
            return x
        pad = self.padded - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rotation. x: [..., dim] -> [..., padded]."""
        x = self._pad(x.to(self.device, self.dtype))
        return (x * self._signs) @ self._H.t()

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse rotation, truncated back to `dim`. y: [..., padded] -> [..., dim]."""
        x = (y @ self._H) * self._signs
        return x[..., : self.dim]


# ---------------------------------------------------------------------------
# TurboQuant (MSE) — per-coordinate uniform scalar quantization
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantConfig:
    bits: int = 4            # bits per coordinate for the MSE stage
    seed: int = 0
    qjl_bits: int = 0        # residual QJL bits for TurboQuantProd (0 = MSE only)


@dataclass
class TurboCode:
    """A compressed vector: integer codes + the per-vector scale + QJL signs."""

    codes: torch.Tensor          # [..., padded] int (0..2^bits-1)
    scale: torch.Tensor          # [..., 1] per-vector quantization scale
    qjl_signs: torch.Tensor | None = None  # [..., qjl_bits] int8 {-1,+1} residual signs
    qjl_resid_norm: torch.Tensor | None = None  # [..., 1] residual L2 norm
    qjl_proj_seed: int = 0


class TurboQuantMSE:
    """MSE-optimized TurboQuant: rotate, then uniform per-coordinate quantization.

    Encodes a vector into integer codes with one shared per-vector scale. The
    rotation concentrates the coordinate distribution so a single symmetric
    uniform quantizer per coordinate is near-optimal for reconstruction MSE.
    """

    def __init__(self, dim: int, config: TurboQuantConfig | None = None,
                 *, device=None, dtype=torch.float32) -> None:
        self.config = config or TurboQuantConfig()
        self.rotation = RandomRotation(dim, seed=self.config.seed, device=device, dtype=dtype)
        self.dim = dim
        self.levels = (1 << self.config.bits) - 1  # e.g. 4 bits -> 15 levels
        self.device = device
        self.dtype = dtype

    def _scale_for(self, rotated: torch.Tensor) -> torch.Tensor:
        # Clip range at a few std of the concentrated rotated coordinates. Using
        # max-abs per vector is simple and keeps the quantizer unbiased in sign.
        return rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    def encode(self, x: torch.Tensor) -> TurboCode:
        rotated = self.rotation.rotate(x)                 # [..., padded]
        scale = self._scale_for(rotated)                  # [..., 1]
        # Map [-scale, scale] -> [0, levels] uniformly, round to nearest integer.
        norm = (rotated / scale).clamp(-1.0, 1.0)         # [-1, 1]
        codes = torch.round((norm + 1.0) / 2.0 * self.levels)
        return TurboCode(codes=codes.to(torch.int32), scale=scale)

    def decode(self, code: TurboCode) -> torch.Tensor:
        norm = code.codes.to(self.dtype) / self.levels * 2.0 - 1.0  # [-1, 1]
        rotated = norm * code.scale
        return self.rotation.unrotate(rotated)            # [..., dim]

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip: encode then decode (reconstruction)."""
        return self.decode(self.encode(x))

    def bits_per_coordinate(self) -> float:
        # Payload bits / original dim: codes cost bits*padded, scale is one fp32.
        return (self.config.bits * self.rotation.padded + 32) / self.dim


# ---------------------------------------------------------------------------
# TurboQuant (prod) — unbiased inner-product via a 1-bit QJL residual
# ---------------------------------------------------------------------------


class TurboQuantProd:
    """Inner-product-optimized TurboQuant: MSE stage + 1-bit QJL residual.

    EXPERIMENTAL / honest result (§16): in BitMem's regime — L2-normalized
    retrieval keys quantized at >= 2 bits per coordinate — the MSE-decoded dot
    product already estimates inner products accurately, and the 1-bit QJL
    residual correction implemented here adds more variance than it removes (see
    the benchmark). The QJL stage is designed to matter at EXTREME compression
    (sub-2-bit, e.g. KV-cache), which is not where the memory store operates.
    `TurboQuantMSE` is therefore the recommended store codec; this class is kept
    for completeness and for the low-bit regime, clearly labeled.

    The MSE quantizer introduces a bias in inner-product estimates. TurboQuantProd
    corrects it by projecting the (rotated-space) residual through a random
    Gaussian matrix and keeping only the SIGN of each projection (1-bit QJL). The
    inner product between two vectors is then estimated as

        <x, y> ~= <decode_mse(x), decode_mse(y)> + c * <qjl_signs(x), qjl_signs(y)>

    where the QJL term is an unbiased estimator of the residual inner product.
    """

    def __init__(self, dim: int, config: TurboQuantConfig | None = None,
                 *, device=None, dtype=torch.float32) -> None:
        cfg = config or TurboQuantConfig(bits=4, qjl_bits=256)
        if cfg.qjl_bits <= 0:
            cfg = TurboQuantConfig(bits=cfg.bits, seed=cfg.seed, qjl_bits=256)
        self.config = cfg
        self.mse = TurboQuantMSE(dim, cfg, device=device, dtype=dtype)
        self.padded = self.mse.rotation.padded
        self.device = device
        self.dtype = dtype
        g = torch.Generator().manual_seed(cfg.seed + 7919)
        # QJL projection in rotated space: [padded, qjl_bits] Gaussian.
        self._P = torch.randn(self.padded, cfg.qjl_bits, generator=g).to(
            device=device, dtype=dtype
        )

    def encode(self, x: torch.Tensor) -> TurboCode:
        code = self.mse.encode(x)
        # Residual in rotated space between the true rotation and the decoded one.
        rotated = self.mse.rotation.rotate(x)
        norm = code.codes.to(self.dtype) / self.mse.levels * 2.0 - 1.0
        decoded_rotated = norm * code.scale
        residual = rotated - decoded_rotated              # [..., padded]
        signs = torch.sign(residual @ self._P)            # [..., qjl_bits]
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        code.qjl_signs = signs.to(torch.int8)
        code.qjl_resid_norm = residual.norm(dim=-1, keepdim=True)
        code.qjl_proj_seed = self.config.seed + 7919
        return code

    def decode(self, code: TurboCode) -> torch.Tensor:
        return self.mse.decode(code)

    def estimate_inner_product(self, a: TurboCode, b: TurboCode) -> torch.Tensor:
        """Unbiased-ish inner-product estimate between two encoded vectors.

        Combines the MSE-decoded dot product with the QJL residual correction.
        Accepts batched codes with matching leading dims (or broadcastable).
        """
        da = self.mse.decode(a)
        db = self.mse.decode(b)
        base = (da * db).sum(dim=-1)
        corr = torch.zeros_like(base)
        if (
            a.qjl_signs is not None and b.qjl_signs is not None
            and a.qjl_resid_norm is not None and b.qjl_resid_norm is not None
        ):
            # 1-bit QJL recovers the ANGLE between the two residuals via the
            # sign-agreement rate (Goemans-Williamson / SimHash identity):
            #   E[sign(<r_a,p>) sign(<r_b,p>)] = 1 - 2*theta/pi
            # so cos(theta) = cos( (pi/2) * (1 - agreement) ), and the residual
            # inner product is ||r_a|| ||r_b|| cos(theta). This recovers the
            # residual geometry the MSE stage dropped — the TurboQuantProd
            # correction (§ paper's two-stage inner-product estimator).
            m = a.qjl_signs.shape[-1]
            agree = (
                a.qjl_signs.to(self.dtype) * b.qjl_signs.to(self.dtype)
            ).sum(dim=-1) / m                               # in [-1, 1]
            theta = (math.pi / 2.0) * (1.0 - agree)         # estimated angle
            cos_resid = torch.cos(theta)
            na = a.qjl_resid_norm.squeeze(-1)
            nb = b.qjl_resid_norm.squeeze(-1)
            corr = na * nb * cos_resid
        return base + corr

    def bits_per_coordinate(self) -> float:
        payload = self.config.bits * self.padded + 32 + self.config.qjl_bits
        return payload / self.mse.dim
