"""Spatial (within-frame) self-attention for video diffusion transformers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from bitvideo.ops import Backend, KernelVariant, WeightLayout
from bitvideo.quantization import QuantizationConfig

from .attention import Attention, KVCache, RMSNorm, _positive_int
from .feedforward import FeedForward
from .rope import RotaryFrequencies


class SpatialAttention(nn.Module):
    """Self-attention applied independently within each video frame.

    Input tokens of shape ``[B, T*H*W, D]`` are reshaped so that each frame's
    spatial tokens attend only to other tokens in the same frame, producing
    within-frame feature mixing without temporal interaction.

    The module wraps the base :class:`Attention` primitive and adds:

    * Automatic reshape from ``[B, T*H*W, D]`` to ``[B*T, H*W, D]`` before
      attention and back afterward.
    * Optional pre-attention layer normalization (default: RMSNorm).
    * Optional post-attention residual connection.
    * Optional post-attention feed-forward network with its own norm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        qkv_bias: bool = True,
        output_bias: bool = True,
        qk_norm: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        is_causal: bool = False,
        use_native_gqa: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1.0e-6,
        residual: bool = True,
        ffn: bool = False,
        ffn_expansion_ratio: float = 4.0,
        ffn_activation: str = "swiglu",
        ffn_dropout: float = 0.0,
        quantization: QuantizationConfig | None = None,
        inference_layout: str | int | WeightLayout = "auto",
        backend: str | Backend = Backend.AUTO,
        variant: int | KernelVariant = KernelVariant.AUTO,
        split_k: int = 0,
        autotune: bool = False,
        use_packed_inference: bool = True,
        auto_pack: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        if not isinstance(residual, bool):
            raise TypeError("residual must be bool")
        if not isinstance(ffn, bool):
            raise TypeError("ffn must be bool")
        self.residual = residual
        self.has_ffn = ffn

        # Normalization before attention.
        self.norm = _make_norm(norm_type, dim, eps=norm_eps, device=device, dtype=dtype)

        # Core attention (self-attention, no cross context).
        linear_kwargs: dict[str, Any] = {
            "quantization": quantization,
            "inference_layout": inference_layout,
            "backend": backend,
            "variant": variant,
            "split_k": split_k,
            "autotune": autotune,
            "use_packed_inference": use_packed_inference,
            "auto_pack": auto_pack,
            "device": device,
            "dtype": dtype,
        }
        self.attention = Attention(
            dim,
            num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            output_bias=output_bias,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            is_causal=is_causal,
            use_native_gqa=use_native_gqa,
            **linear_kwargs,
        )

        # Optional FFN block with its own pre-norm.
        if self.has_ffn:
            self.ffn_norm = _make_norm(norm_type, dim, eps=norm_eps, device=device, dtype=dtype)
            self.ffn = FeedForward(
                dim,
                expansion_ratio=ffn_expansion_ratio,
                activation=ffn_activation,
                dropout=ffn_dropout,
                **linear_kwargs,
            )
        else:
            self.ffn_norm = nn.Identity()
            self.ffn = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        temporal_size: int,
        spatial_size: int,
        attention_mask: torch.Tensor | None = None,
        rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply spatial self-attention within each frame.

        Args:
            hidden_states: Token embeddings ``[B, T*H*W, D]``.
            temporal_size: Number of temporal positions ``T``.
            spatial_size: Number of spatial positions per frame ``H*W``.
            attention_mask: Optional mask broadcastable to ``[B*T, Hq, H*W, H*W]``.
            rotary: Optional spatial RoPE frequencies for positions within a frame.

        Returns:
            Tensor of shape ``[B, T*H*W, D]``.
        """

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states must have shape [B, T*H*W, {self.dim}]; "
                f"got {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, _ = hidden_states.shape
        temporal_size = _positive_int(temporal_size, "temporal_size")
        spatial_size = _positive_int(spatial_size, "spatial_size")
        if temporal_size * spatial_size != seq_len:
            raise ValueError(
                f"temporal_size * spatial_size must equal sequence length {seq_len}; "
                f"got {temporal_size} * {spatial_size} = {temporal_size * spatial_size}"
            )

        # residual: [B, T*H*W, D].
        residual_connection = hidden_states

        # Reshape to per-frame sequences: [B*T, H*W, D].
        # hidden_states: [B, T, H*W, D] -> [B*T, H*W, D].
        x = hidden_states.reshape(batch_size * temporal_size, spatial_size, self.dim)

        # Pre-norm: [B*T, H*W, D].
        x = self.norm(x)

        # Spatial self-attention: [B*T, H*W, D].
        x = self.attention(x, attention_mask=attention_mask, rotary=rotary)

        # Reshape back: [B, T*H*W, D].
        x = x.reshape(batch_size, seq_len, self.dim)

        # Residual connection.
        if self.residual:
            x = x + residual_connection

        # Optional FFN with residual.
        if self.has_ffn:
            ffn_residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + ffn_residual

        return x

    @torch.no_grad()
    def pack_weights(
        self,
        layout: int | WeightLayout | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> None:
        """Pack all internal projection weights for inference."""

        self.attention.pack_weights(layout, backend=backend)
        if self.has_ffn and isinstance(self.ffn, FeedForward):
            self.ffn.pack_weights(layout, backend=backend)

    def clear_packed_cache(self) -> None:
        """Discard all derived packed storage."""

        self.attention.clear_packed_cache()
        if self.has_ffn and isinstance(self.ffn, FeedForward):
            self.ffn.clear_packed_cache()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"residual={self.residual}, ffn={self.has_ffn}"
        )


def _make_norm(
    norm_type: str,
    dim: int,
    *,
    eps: float = 1.0e-6,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Create a normalization layer by name."""

    normalized = str(norm_type).strip().lower().replace("-", "_")
    if normalized in {"rmsnorm", "rms_norm", "rms"}:
        return RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    if normalized in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(
        f"norm_type must be one of rmsnorm, layernorm, identity; got {norm_type!r}"
    )
