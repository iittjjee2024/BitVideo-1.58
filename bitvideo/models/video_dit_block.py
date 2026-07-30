"""Single transformer block for the BitVideo-1.58 Video DiT architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from bitvideo.ops import Backend, KernelVariant, WeightLayout
from bitvideo.quantization import BitLinear, QuantizationConfig

from .attention import KVCache, RMSNorm, _positive_int, _probability
from .cross_attention import CrossAttention
from .feedforward import FeedForward
from .rope import RotaryFrequencies
from .spatial_attention import SpatialAttention, _make_norm
from .temporal_attention import TemporalAttention


class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization conditioned on a timestep embedding.

    Produces per-sample shift and scale modulation vectors from an external
    conditioning signal (typically the diffusion timestep embedding), following
    the DiT adaptive normalization pattern.
    """

    def __init__(
        self,
        dim: int,
        *,
        conditioning_dim: int | None = None,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.conditioning_dim = dim if conditioning_dim is None else _positive_int(
            conditioning_dim, "conditioning_dim"
        )
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        # Linear projects conditioning to shift and scale: [conditioning_dim] -> [2*dim].
        self.linear = nn.Linear(
            self.conditioning_dim,
            2 * dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization.

        Args:
            x: Input tensor ``[B, L, D]``.
            conditioning: Conditioning vector ``[B, conditioning_dim]``.

        Returns:
            Modulated tensor ``[B, L, D]``.
        """

        if not isinstance(x, torch.Tensor) or not isinstance(conditioning, torch.Tensor):
            raise TypeError("x and conditioning must be torch.Tensor instances")
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, L, {self.dim}]; got {tuple(x.shape)}")
        if conditioning.ndim != 2 or conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"conditioning must have shape [B, {self.conditioning_dim}]; "
                f"got {tuple(conditioning.shape)}"
            )
        if conditioning.shape[0] != x.shape[0]:
            raise ValueError("conditioning batch size must match x")
        # modulation: [B, 2*D].
        modulation = self.linear(conditioning)
        # shift/scale: [B, 1, D] each.
        shift, scale = modulation.chunk(2, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        # normalized: [B, L, D]; output: [B, L, D].
        normalized = self.norm(x)
        return normalized * (1.0 + scale) + shift

    def extra_repr(self) -> str:
        return f"dim={self.dim}, conditioning_dim={self.conditioning_dim}, eps={self.eps:g}"


class AdaLayerNormZero(nn.Module):
    """Adaptive normalization with zero-initialized gate for DiT blocks.

    Produces shift, scale, and gate vectors for the attention and FFN paths,
    following the DiT-XL "adaLN-Zero" pattern where gates start at zero for
    stable early training.
    """

    def __init__(
        self,
        dim: int,
        *,
        conditioning_dim: int | None = None,
        num_modulations: int = 6,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.conditioning_dim = dim if conditioning_dim is None else _positive_int(
            conditioning_dim, "conditioning_dim"
        )
        self.num_modulations = _positive_int(num_modulations, "num_modulations")
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        # Projects conditioning to all modulation vectors at once.
        self.linear = nn.Linear(
            self.conditioning_dim,
            num_modulations * dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute normalized input and modulation parameters.

        Args:
            x: Input tensor ``[B, L, D]``.
            conditioning: Conditioning vector ``[B, conditioning_dim]``.

        Returns:
            Tuple of (normalized_x ``[B, L, D]``, list of modulation tensors
            each ``[B, 1, D]``).
        """

        if not isinstance(x, torch.Tensor) or not isinstance(conditioning, torch.Tensor):
            raise TypeError("x and conditioning must be torch.Tensor instances")
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, L, {self.dim}]; got {tuple(x.shape)}")
        if conditioning.ndim != 2 or conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"conditioning must have shape [B, {self.conditioning_dim}]; "
                f"got {tuple(conditioning.shape)}"
            )
        if conditioning.shape[0] != x.shape[0]:
            raise ValueError("conditioning batch size must match x")
        # modulation: [B, num_modulations*D].
        modulation = self.linear(conditioning)
        # chunks: list of num_modulations tensors, each [B, D] -> [B, 1, D].
        chunks = [chunk.unsqueeze(1) for chunk in modulation.chunk(self.num_modulations, dim=-1)]
        # normalized: [B, L, D].
        normalized = self.norm(x)
        return normalized, chunks

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, conditioning_dim={self.conditioning_dim}, "
            f"num_modulations={self.num_modulations}, eps={self.eps:g}"
        )


class VideoDiTBlock(nn.Module):
    """Single Video Diffusion Transformer block.

    Combines spatial self-attention, temporal self-attention, cross-attention
    for text conditioning, and a feed-forward network. Each sub-layer uses
    adaptive normalization conditioned on the diffusion timestep embedding.

    Architecture per block:
        1. AdaLN-Zero → Spatial self-attention → gate → residual
        2. AdaLN-Zero → Temporal self-attention → gate → residual
        3. AdaLN → Cross-attention → residual
        4. AdaLN-Zero → Feed-forward → gate → residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        context_dim: int | None = None,
        conditioning_dim: int | None = None,
        ffn_expansion_ratio: float = 4.0,
        ffn_activation: str = "swiglu",
        qkv_bias: bool = True,
        output_bias: bool = True,
        qk_norm: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_native_gqa: bool = True,
        temporal_causal: bool = False,
        norm_eps: float = 1.0e-6,
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
        self.context_dim = dim if context_dim is None else _positive_int(
            context_dim, "context_dim"
        )
        self.conditioning_dim = dim if conditioning_dim is None else _positive_int(
            conditioning_dim, "conditioning_dim"
        )
        self.temporal_causal = bool(temporal_causal)

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
        attention_kwargs: dict[str, Any] = {
            "head_dim": head_dim,
            "num_kv_heads": num_kv_heads,
            "qkv_bias": qkv_bias,
            "output_bias": output_bias,
            "qk_norm": qk_norm,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "use_native_gqa": use_native_gqa,
            **linear_kwargs,
        }

        # 1. Spatial self-attention with adaLN-Zero (shift1, scale1, gate1).
        self.spatial_norm = AdaLayerNormZero(
            dim,
            conditioning_dim=self.conditioning_dim,
            num_modulations=3,
            eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        self.spatial_attention = SpatialAttention(
            dim,
            num_heads,
            norm_type="identity",
            residual=False,
            **attention_kwargs,
        )

        # 2. Temporal self-attention with adaLN-Zero (shift2, scale2, gate2).
        self.temporal_norm = AdaLayerNormZero(
            dim,
            conditioning_dim=self.conditioning_dim,
            num_modulations=3,
            eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        self.temporal_attention = TemporalAttention(
            dim,
            num_heads,
            is_causal=temporal_causal,
            norm_type="identity",
            residual=False,
            **attention_kwargs,
        )

        # 3. Cross-attention with simple adaLN (shift, scale).
        self.cross_norm = AdaLayerNorm(
            dim,
            conditioning_dim=self.conditioning_dim,
            eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        self.cross_attention = CrossAttention(
            dim,
            num_heads,
            context_dim=self.context_dim,
            norm_type="identity",
            residual=False,
            gate=True,
            **attention_kwargs,
        )

        # 4. Feed-forward with adaLN-Zero (shift3, scale3, gate3).
        self.ffn_norm = AdaLayerNormZero(
            dim,
            conditioning_dim=self.conditioning_dim,
            num_modulations=3,
            eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        self.ffn = FeedForward(
            dim,
            expansion_ratio=ffn_expansion_ratio,
            activation=ffn_activation,
            dropout=ffn_dropout,
            **linear_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor,
        temporal_size: int,
        spatial_size: int,
        context: torch.Tensor | None = None,
        context_cache: KVCache | None = None,
        spatial_rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        temporal_rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        spatial_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        return_context_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Process one transformer block.

        Args:
            hidden_states: Token embeddings ``[B, T*H*W, D]``.
            timestep_embedding: Diffusion step conditioning ``[B, conditioning_dim]``.
            temporal_size: Number of frames ``T``.
            spatial_size: Spatial tokens per frame ``H*W``.
            context: Text conditioning ``[B, Lc, context_dim]`` (first call only).
            context_cache: Cached cross-attention KV from a previous call.
            spatial_rotary: Spatial RoPE frequencies.
            temporal_rotary: Temporal RoPE frequencies.
            spatial_mask: Spatial attention mask.
            temporal_mask: Temporal attention mask.
            cross_mask: Cross-attention mask.
            return_context_cache: Whether to return the cross-attention cache.

        Returns:
            Output ``[B, T*H*W, D]``, optionally with cross-attention cache.
        """

        # --- 1. Spatial self-attention ---
        # normalized_spatial: [B, T*H*W, D]; [shift, scale, gate]: each [B, 1, D].
        normalized_spatial, spatial_mods = self.spatial_norm(hidden_states, timestep_embedding)
        shift_s, scale_s, gate_s = spatial_mods[0], spatial_mods[1], spatial_mods[2]
        # Modulate: [B, T*H*W, D].
        modulated_spatial = normalized_spatial * (1.0 + scale_s) + shift_s
        # Attention: [B, T*H*W, D].
        spatial_out = self.spatial_attention(
            modulated_spatial,
            temporal_size=temporal_size,
            spatial_size=spatial_size,
            attention_mask=spatial_mask,
            rotary=spatial_rotary,
        )
        # Gated residual: [B, T*H*W, D].
        hidden_states = hidden_states + torch.tanh(gate_s) * spatial_out

        # --- 2. Temporal self-attention ---
        # normalized_temporal: [B, T*H*W, D]; [shift, scale, gate]: each [B, 1, D].
        normalized_temporal, temporal_mods = self.temporal_norm(hidden_states, timestep_embedding)
        shift_t, scale_t, gate_t = temporal_mods[0], temporal_mods[1], temporal_mods[2]
        # Modulate: [B, T*H*W, D].
        modulated_temporal = normalized_temporal * (1.0 + scale_t) + shift_t
        # Attention: [B, T*H*W, D].
        temporal_out = self.temporal_attention(
            modulated_temporal,
            temporal_size=temporal_size,
            spatial_size=spatial_size,
            attention_mask=temporal_mask,
            rotary=temporal_rotary,
        )
        # Gated residual: [B, T*H*W, D].
        hidden_states = hidden_states + torch.tanh(gate_t) * temporal_out

        # --- 3. Cross-attention ---
        # cross_normalized: [B, T*H*W, D].
        cross_normalized = self.cross_norm(hidden_states, timestep_embedding)
        if context is not None:
            # Ensure context matches query dtype (important under autocast).
            context_matched = context.to(dtype=cross_normalized.dtype)
            cross_out, cross_kv_cache = self.cross_attention(
                cross_normalized,
                context=context_matched,
                attention_mask=cross_mask,
                return_cache=True,
            )
        else:
            cross_result = self.cross_attention(
                cross_normalized,
                context_cache=context_cache,
                attention_mask=cross_mask,
                return_cache=return_context_cache,
            )
            if return_context_cache:
                cross_out, cross_kv_cache = cross_result
            else:
                cross_out = cross_result
                cross_kv_cache = context_cache
        # Residual (gate is inside CrossAttention): [B, T*H*W, D].
        hidden_states = hidden_states + cross_out

        # --- 4. Feed-forward ---
        # normalized_ffn: [B, T*H*W, D]; [shift, scale, gate]: each [B, 1, D].
        normalized_ffn, ffn_mods = self.ffn_norm(hidden_states, timestep_embedding)
        shift_f, scale_f, gate_f = ffn_mods[0], ffn_mods[1], ffn_mods[2]
        # Modulate: [B, T*H*W, D].
        modulated_ffn = normalized_ffn * (1.0 + scale_f) + shift_f
        # FFN: [B, T*H*W, D].
        ffn_out = self.ffn(modulated_ffn)
        # Gated residual: [B, T*H*W, D].
        hidden_states = hidden_states + torch.tanh(gate_f) * ffn_out

        if return_context_cache:
            return hidden_states, cross_kv_cache
        return hidden_states

    @torch.no_grad()
    def pack_weights(
        self,
        layout: int | WeightLayout | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> None:
        """Pack all projection weights for inference."""

        self.spatial_attention.pack_weights(layout, backend=backend)
        self.temporal_attention.pack_weights(layout, backend=backend)
        self.cross_attention.pack_weights(layout, backend=backend)
        self.ffn.pack_weights(layout, backend=backend)

    def clear_packed_cache(self) -> None:
        """Discard all packed weight caches."""

        self.spatial_attention.clear_packed_cache()
        self.temporal_attention.clear_packed_cache()
        self.cross_attention.clear_packed_cache()
        self.ffn.clear_packed_cache()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"context_dim={self.context_dim}, conditioning_dim={self.conditioning_dim}, "
            f"temporal_causal={self.temporal_causal}"
        )
