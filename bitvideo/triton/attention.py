"""Triton Flash Attention kernel for memory-efficient scaled dot-product attention."""

from __future__ import annotations

import math

import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


if _TRITON_AVAILABLE:
    @triton.jit
    def _flash_attention_fwd_kernel(
        # Pointers.
        q_ptr, k_ptr, v_ptr, out_ptr,
        # Dimensions.
        B, H, L, D,
        # Strides for Q: [B, H, L, D].
        stride_qb, stride_qh, stride_ql, stride_qd,
        # Strides for K: [B, H, L, D].
        stride_kb, stride_kh, stride_kl, stride_kd,
        # Strides for V: [B, H, L, D].
        stride_vb, stride_vh, stride_vl, stride_vd,
        # Strides for Out: [B, H, L, D].
        stride_ob, stride_oh, stride_ol, stride_od,
        # Scale factor.
        sm_scale,
        # Block sizes.
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Simplified flash attention forward kernel.

        Computes online softmax over the full sequence for each query block,
        accumulating attention output in a numerically stable fashion using
        the log-sum-exp trick.
        """
        # Which batch and head this program handles.
        pid_bh = tl.program_id(0)
        pid_l = tl.program_id(1)
        batch_idx = pid_bh // H
        head_idx = pid_bh % H

        # Query offsets for this tile: [BLOCK_L, D].
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, BLOCK_D)

        # Load query tile: [BLOCK_L, D].
        q_ptrs = (q_ptr + batch_idx * stride_qb + head_idx * stride_qh
                  + offs_l[:, None] * stride_ql + offs_d[None, :] * stride_qd)
        mask_q = (offs_l[:, None] < L) & (offs_d[None, :] < D)
        q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)
        q = q * sm_scale

        # Online softmax accumulators.
        m_i = tl.full((BLOCK_L,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_L,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)

        # Iterate over all key/value blocks.
        for k_start in range(0, L, BLOCK_L):
            offs_kl = k_start + tl.arange(0, BLOCK_L)

            # Load key tile: [BLOCK_L, D].
            k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh
                      + offs_kl[:, None] * stride_kl + offs_d[None, :] * stride_kd)
            mask_k = (offs_kl[:, None] < L) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

            # QK^T: [BLOCK_L, BLOCK_L].
            qk = tl.dot(q, tl.trans(k))
            # Mask out-of-bounds.
            mask_qk = offs_kl[None, :] < L
            qk = tl.where(mask_qk, qk, float("-inf"))

            # Online softmax update.
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # Load value tile: [BLOCK_L, D].
            v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh
                      + offs_kl[:, None] * stride_vl + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptrs, mask=mask_k, other=0.0).to(tl.float32)

            # Accumulate: [BLOCK_L, D].
            acc += tl.dot(p, v)
            m_i = m_new

        # Normalize by sum of exponentials.
        acc = acc / l_i[:, None]

        # Store output: [BLOCK_L, D].
        out_ptrs = (out_ptr + batch_idx * stride_ob + head_idx * stride_oh
                    + offs_l[:, None] * stride_ol + offs_d[None, :] * stride_od)
        mask_out = (offs_l[:, None] < L) & (offs_d[None, :] < D)
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


def triton_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Triton Flash Attention forward pass.

    Args:
        query: Query tensor ``[B, H, L, D]``.
        key: Key tensor ``[B, H, L, D]``.
        value: Value tensor ``[B, H, L, D]``.
        scale: Softmax scale (default: 1/sqrt(D)).

    Returns:
        Attention output ``[B, H, L, D]``.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")

    B, H, L, D = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.empty_like(query)

    BLOCK_L = min(64, L)
    BLOCK_D = D  # Assume D fits in a single tile.
    grid = (B * H, (L + BLOCK_L - 1) // BLOCK_L)

    _flash_attention_fwd_kernel[grid](
        query, key, value, output,
        B, H, L, D,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
    )
    return output


if not _TRITON_AVAILABLE:
    def triton_flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Fallback when Triton is unavailable."""
        raise RuntimeError("Triton is not installed; cannot use triton_flash_attention")
