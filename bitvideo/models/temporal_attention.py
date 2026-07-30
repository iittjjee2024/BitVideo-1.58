"""Temporal (across-frame) self-attention for video diffusion transformers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from bitvideo.ops import Backend, KernelVariant, WeightLayout
from bitvideo.quantization import QuantizationConfig

from .attention import Attention, KVCache, _positive_int
from .feedforward import FeedForward
from .rope import RotaryFrequencies
from .spatial_attention import _make_norm


class TemporalAttention(nn.Module):
    """Self-attention applied independently across frames at each spatial position.

    Input tokens of shape ``[B, T*H*W, D]`` are reshaped so that each spatial
    position's temporal sequence attends only to the same position across
    frames, producing temporal feature mixing without spatial interaction.

    The module wraps the base :class:`Attention` primitive and adds:

    * Automatic reshape from ``[B, T*H*W, D]`` to ``[B*H*W, T, D]`` before
      attention and back afterward.
    * Optional pre-attention layer normalization (default: RMSNorm).
    * Optional causal masking for autoregressive temporal generation.
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
        self.is_causal = is_causal

        # Normalization before attention.
        self.norm = _make_norm(norm_type, dim, eps=norm_eps, device=device, dtype=dtype)

        # Core attention (self-attention across temporal positions).
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
        past_key_value: KVCache | tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Apply temporal self-attention across frames.

        Args:
            hidden_states: Token embeddings ``[B, T*H*W, D]``.
            temporal_size: Number of temporal positions ``T``.
            spatial_size: Number of spatial positions per frame ``H*W``.
            attention_mask: Optional mask broadcastable to ``[B*H*W, Hq, T, T]``.
            rotary: Optional temporal RoPE frequencies for positions across frames.
            past_key_value: Optional KV cache for autoregressive temporal decoding.
            use_cache: Whether to return the updated KV cache.

        Returns:
            Output tensor ``[B, T*H*W, D]``, optionally with KV cache.
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

        # Reshape to per-position temporal sequences.
        # hidden_states: [B, T, H*W, D] -> [B, H*W, T, D] -> [B*H*W, T, D].
        x = hidden_states.reshape(batch_size, temporal_size, spatial_size, self.dim)
        # x: [B, T, H*W, D] -> transpose to [B, H*W, T, D].
        x = x.permute(0, 2, 1, 3).reshape(batch_size * spatial_size, temporal_size, self.dim)

        # Pre-norm: [B*H*W, T, D].
        x = self.norm(x)

        # Temporal self-attention: [B*H*W, T, D], with optional cache.
        attention_result = self.attention(
            x,
            attention_mask=attention_mask,
            rotary=rotary,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            x, cache = attention_result
        else:
            x = attention_result
            cache = None

        # Reshape back: [B*H*W, T, D] -> [B, H*W, T, D] -> [B, T, H*W, D] -> [B, T*H*W, D].
        x = x.reshape(batch_size, spatial_size, temporal_size, self.dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)

        # Residual connection.
        if self.residual:
            x = x + residual_connection

        # Optional FFN with residual.
        if self.has_ffn:
            ffn_residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + ffn_residual

        if use_cache:
            return x, cache
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
            f"is_causal={self.is_causal}, residual={self.residual}, ffn={self.has_ffn}"
        )
