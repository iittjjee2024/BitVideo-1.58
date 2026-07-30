"""Cross-attention for text/image conditioning in video diffusion transformers."""

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


class CrossAttention(nn.Module):
    """Cross-attention for conditioning video tokens on external context.

    Wraps the base :class:`Attention` with ``static_kv=True`` semantics for
    efficient text/image conditioning. The conditioning context is projected
    once and cached; subsequent calls reuse the cache without reprojecting.

    This is the standard cross-attention pattern used in diffusion transformers
    where text encoder outputs condition every denoising step.

    Features:

    * Automatic static KV cache management: builds on first call with context,
      reuses on subsequent calls.
    * Separate ``context_dim`` for conditioning from a different embedding space.
    * Optional pre-attention normalization on both query and context.
    * Optional post-attention residual connection.
    * Optional post-attention feed-forward network with its own norm.
    * Optional gate for adaptive conditioning strength (zero-initialized).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        context_dim: int | None = None,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        qkv_bias: bool = True,
        output_bias: bool = True,
        qk_norm: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        use_native_gqa: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1.0e-6,
        context_norm: bool = False,
        residual: bool = True,
        gate: bool = False,
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
        self.context_dim = dim if context_dim is None else _positive_int(
            context_dim, "context_dim"
        )
        if not isinstance(residual, bool):
            raise TypeError("residual must be bool")
        if not isinstance(gate, bool):
            raise TypeError("gate must be bool")
        if not isinstance(ffn, bool):
            raise TypeError("ffn must be bool")
        if not isinstance(context_norm, bool):
            raise TypeError("context_norm must be bool")
        self.residual = residual
        self.has_gate = gate
        self.has_ffn = ffn

        # Normalization before attention (applied to query tokens).
        self.norm = _make_norm(norm_type, dim, eps=norm_eps, device=device, dtype=dtype)

        # Optional normalization for the conditioning context.
        self.context_norm_layer = (
            _make_norm(norm_type, self.context_dim, eps=norm_eps, device=device, dtype=dtype)
            if context_norm
            else nn.Identity()
        )

        # Core cross-attention with context_dim for key/value projections.
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
            context_dim=self.context_dim,
            qkv_bias=qkv_bias,
            output_bias=output_bias,
            qk_norm=qk_norm,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            is_causal=False,
            use_native_gqa=use_native_gqa,
            **linear_kwargs,
        )

        # Optional learnable gate (zero-initialized for stable early training).
        if self.has_gate:
            self.gate_param = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("gate_param", None)

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
        context: torch.Tensor | None = None,
        context_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Apply cross-attention conditioning.

        On the first call, ``context`` must be provided to build the static KV
        cache. On subsequent calls within the same conditioning, pass
        ``context_cache`` instead and omit ``context`` for zero-cost reuse.

        Args:
            hidden_states: Query tokens ``[B, L, D]``.
            context: Conditioning context ``[B, Lc, context_dim]`` for initial projection.
            context_cache: Pre-computed KV cache from a previous call.
            attention_mask: Optional attention mask broadcastable to ``[B, H, L, Lc]``.
            key_padding_mask: Optional key padding mask ``[B, Lc]``.
            rotary: Optional query RoPE frequencies.
            key_rotary: Optional key RoPE frequencies (only valid on initial projection).
            return_cache: Whether to return the KV cache for later reuse.

        Returns:
            Output tensor ``[B, L, D]``, optionally with the static KV cache.
        """

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states must have shape [B, L, {self.dim}]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if context is None and context_cache is None:
            raise ValueError("either context or context_cache must be provided")
        if context is not None and context_cache is not None:
            raise ValueError("context and context_cache are mutually exclusive")

        # residual: [B, L, D].
        residual_connection = hidden_states

        # Pre-norm on query: [B, L, D].
        x = self.norm(hidden_states)

        # Determine whether we're building or reusing the cache.
        if context is not None:
            # Normalize context if configured: [B, Lc, context_dim].
            normalized_context = self.context_norm_layer(context)
            # Match dtype to query (critical under autocast where norms may cast).
            normalized_context = normalized_context.to(dtype=x.dtype)

            # Initial call: project context and build cache.
            # output: [B, L, D]; cache returned via use_cache.
            output, cache = self.attention(
                x,
                context=normalized_context,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                rotary=rotary,
                key_rotary=key_rotary,
                static_kv=True,
                use_cache=True,
            )
        else:
            # Reuse call: skip context projection entirely.
            # output: [B, L, D]; cache is passthrough.
            attention_result = self.attention(
                x,
                past_key_value=context_cache,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                rotary=rotary,
                static_kv=True,
                use_cache=return_cache,
            )
            if return_cache:
                output, cache = attention_result
            else:
                output = attention_result
                cache = context_cache

        # Apply gate if configured: [B, L, D].
        if self.has_gate:
            # gate_param: [D]; tanh activation keeps values bounded.
            output = output * torch.tanh(self.gate_param)

        # Residual connection.
        if self.residual:
            x = output + residual_connection
        else:
            x = output

        # Optional FFN with residual.
        if self.has_ffn:
            ffn_residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + ffn_residual

        if return_cache:
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
            f"dim={self.dim}, context_dim={self.context_dim}, num_heads={self.num_heads}, "
            f"residual={self.residual}, gate={self.has_gate}, ffn={self.has_ffn}"
        )
