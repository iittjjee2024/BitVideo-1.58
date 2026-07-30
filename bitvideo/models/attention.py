"""Quantized multi-head attention built on PyTorch scaled-dot-product attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitvideo.ops import Backend, KernelVariant, PackedTernaryWeight, WeightLayout
from bitvideo.quantization import BitLinear, QuantizationConfig

from .rope import RotaryFrequencies, apply_rotary_embedding, apply_rotary_qk


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


@dataclass(frozen=True)
class KVCache:
    """Projected key/value cache with tensors shaped ``[B, Hkv, L, Dh]``.

    Entries hold attention keys and values that are ready to use. Keys have
    already passed through ``Attention.key_norm`` (an identity when the module
    was built with ``qk_norm=False``) and already carry whatever rotary
    rotation the producing call applied. Values are projected only, since
    neither normalization nor rotation applies to them. Consumers must not
    normalize or rotate cached keys a second time, and this container does not
    record which frequencies were used.
    """

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.key, torch.Tensor) or not isinstance(self.value, torch.Tensor):
            raise TypeError("key and value must be torch.Tensor instances")
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("key and value must have shape [B,Hkv,L,Dh]")
        if self.key.shape != self.value.shape:
            raise ValueError(
                f"key and value shapes must match; got {tuple(self.key.shape)} and "
                f"{tuple(self.value.shape)}"
            )
        if any(extent <= 0 for extent in self.key.shape[1:]):
            raise ValueError("cache head, sequence, and head-dimension extents must be positive")
        if not self.key.is_floating_point() or self.value.dtype != self.key.dtype:
            raise TypeError("key and value must share a floating-point dtype")
        if self.key.device != self.value.device:
            raise ValueError("key and value must reside on the same device")

    @property
    def batch_size(self) -> int:
        return self.key.shape[0]

    @property
    def num_heads(self) -> int:
        return self.key.shape[1]

    @property
    def sequence_length(self) -> int:
        return self.key.shape[2]

    @property
    def head_dim(self) -> int:
        return self.key.shape[3]

    def append(self, key: torch.Tensor, value: torch.Tensor) -> "KVCache":
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("appended key and value must share shape [B,Hkv,L,Dh]")
        if key.shape[0] != self.batch_size:
            raise ValueError("appended cache batch size does not match")
        if key.shape[1] != self.num_heads or key.shape[3] != self.head_dim:
            raise ValueError("appended cache head geometry does not match")
        if key.shape[2] <= 0:
            raise ValueError("appended cache sequence length must be positive")
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise TypeError("appended cache dtype does not match")
        if key.device != self.key.device or value.device != self.value.device:
            raise ValueError("appended cache device does not match")
        # combined_key/value: [B,Hkv,Lpast+Lnew,Dh].
        combined_key = torch.cat((self.key, key), dim=-2)
        combined_value = torch.cat((self.value, value), dim=-2)
        return KVCache(combined_key, combined_value)

    def detach(self) -> "KVCache":
        # Detached tensors preserve [B,Hkv,L,Dh].
        return KVCache(self.key.detach(), self.value.detach())

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "KVCache":
        target_dtype = self.key.dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype) or not target_dtype.is_floating_point:
            raise TypeError("cache dtype must be floating point")
        return KVCache(
            self.key.to(device=device, dtype=target_dtype, non_blocking=non_blocking),
            self.value.to(device=device, dtype=target_dtype, non_blocking=non_blocking),
        )


def _flatten_cache(cache: KVCache) -> tuple[list[torch.Tensor], None]:
    return [cache.key, cache.value], None


def _unflatten_cache(tensors: list[torch.Tensor], context: None) -> KVCache:
    del context
    return KVCache(tensors[0], tensors[1])


try:
    from torch.utils import _pytree

    _pytree.register_pytree_node(KVCache, _flatten_cache, _unflatten_cache)
except (ImportError, ValueError):
    pass


class RMSNorm(nn.Module):
    """Root-mean-square normalization with float32 reduction."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1.0e-6,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        if not isinstance(elementwise_affine, bool):
            raise TypeError("elementwise_affine must be bool")
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating-point torch.Tensor")
        if x.shape[-1] != self.dim:
            raise ValueError(f"x.shape[-1] must be {self.dim}; got {x.shape[-1]}")
        # mean_square/inverse_rms: x.shape[:-1] + [1], accumulated in float32.
        mean_square = x.to(torch.float32).square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps).to(x.dtype)
        # normalized: x.shape in the projection storage dtype.
        normalized = x * inverse_rms
        if self.weight is None:
            return normalized
        return normalized * self.weight.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}"


def _canonical_attention_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor or None")
    if mask.device != device:
        raise ValueError(f"attention_mask must be on {device}; got {mask.device}")
    if mask.dtype is not torch.bool and not mask.is_floating_point():
        raise TypeError("attention_mask must use bool or a floating-point dtype")
    if mask.ndim == 2:
        if mask.shape[0] not in {1, query_length} or mask.shape[1] not in {1, key_length}:
            raise ValueError(
                f"2D attention_mask must broadcast to [{query_length},{key_length}]; "
                f"got {tuple(mask.shape)}"
            )
        # canonical: [1,1,Lq,Lk].
        canonical = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[0] in {1, batch_size}:
            # canonical: [B,1,Lq,Lk].
            canonical = mask.unsqueeze(1)
        elif mask.shape[0] == batch_size * num_heads:
            # canonical: [B,Hq,Lq,Lk].
            canonical = mask.reshape(batch_size, num_heads, mask.shape[-2], mask.shape[-1])
        else:
            raise ValueError("3D attention_mask leading extent must be B or B*Hq")
    elif mask.ndim == 4:
        canonical = mask
    else:
        raise ValueError("attention_mask rank must be 2, 3, or 4")
    target_shape = (batch_size, num_heads, query_length, key_length)
    try:
        torch.broadcast_shapes(tuple(canonical.shape), target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"attention_mask shape {tuple(canonical.shape)} cannot broadcast to {target_shape}"
        ) from exc
    return canonical if canonical.dtype is torch.bool else canonical.to(dtype=dtype)


def _merge_masks(
    first: torch.Tensor | None,
    second: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    if first.dtype is torch.bool and second.dtype is torch.bool:
        # merged bool mask: broadcasted [B,H,Lq,Lk], True means allowed.
        return first & second
    negative_infinity = torch.tensor(float("-inf"), dtype=dtype, device=first.device)
    first_additive = (
        torch.zeros_like(first, dtype=dtype).masked_fill(~first, negative_infinity)
        if first.dtype is torch.bool
        else first.to(dtype)
    )
    second_additive = (
        torch.zeros_like(second, dtype=dtype).masked_fill(~second, negative_infinity)
        if second.dtype is torch.bool
        else second.to(dtype)
    )
    # merged additive mask: broadcasted [B,H,Lq,Lk].
    return first_additive + second_additive


def _padding_mask(
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    if not isinstance(key_padding_mask, torch.Tensor):
        raise TypeError("key_padding_mask must be a torch.Tensor or None")
    if key_padding_mask.shape != (batch_size, key_length):
        raise ValueError(
            f"key_padding_mask must have shape [{batch_size},{key_length}]; "
            f"got {tuple(key_padding_mask.shape)}"
        )
    if key_padding_mask.device != device:
        raise ValueError(f"key_padding_mask must be on {device}; got {key_padding_mask.device}")
    if key_padding_mask.dtype is torch.bool:
        # keep: [B,1,1,Lk], standard key-padding True entries are disallowed.
        return (~key_padding_mask).view(batch_size, 1, 1, key_length)
    if not key_padding_mask.is_floating_point():
        raise TypeError("key_padding_mask must use bool or a floating-point dtype")
    # additive: [B,1,1,Lk].
    return key_padding_mask.to(dtype).view(batch_size, 1, 1, key_length)


def _causal_mask(
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    # Bottom-right alignment uses the signed key/query length difference.
    query_offset = key_length - query_length
    # query_positions: [Lq,1]; key_positions: [1,Lk].
    query_positions = torch.arange(query_length, device=device).view(query_length, 1)
    key_positions = torch.arange(key_length, device=device).view(1, key_length)
    # allowed: [1,1,Lq,Lk].
    return (key_positions <= query_positions + query_offset).view(
        1,
        1,
        query_length,
        key_length,
    )


class Attention(nn.Module):
    """Self/cross attention with BitLinear projections and portable SDPA dispatch.

    Boolean ``attention_mask`` values follow SDPA semantics (``True`` means
    allowed). Boolean ``key_padding_mask`` follows ``nn.MultiheadAttention``
    semantics (``True`` means padding and is therefore disallowed).

    Two distinct caching modes share the ``past_key_value``/``use_cache``
    arguments and are selected by ``static_kv``:

    ``static_kv=False`` (default) grows the cache. Each call projects its
    context and appends the result, which is the incremental self-attention
    behavior used for autoregressive decoding.

    ``static_kv=True`` reuses a fixed cache, which is the cross-attention
    behavior used when the same conditioning context is attended at every
    step. The first call must supply ``context``, and with ``use_cache=True``
    it returns the cache built from that context. Subsequent calls must omit
    ``context`` and pass that cache; they skip the key/value projections
    entirely and leave the cache length unchanged.

    Cached keys are stored after ``key_norm`` and after whichever rotation the
    building call applied, which is ``key_rotary`` when it was supplied and
    ``rotary`` otherwise. Reuse calls therefore reject ``key_rotary`` and
    rotate the query only. Passing a ``rotary`` argument consistent with the
    building call remains the caller's responsibility, because the cached
    tensors alone cannot reveal which frequencies produced them.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        context_dim: int | None = None,
        qkv_bias: bool = True,
        output_bias: bool = True,
        qk_norm: bool = False,
        qk_norm_eps: float = 1.0e-6,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        use_native_gqa: bool = True,
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
        if head_dim is None:
            if self.dim % self.num_heads:
                raise ValueError("dim must be divisible by num_heads when head_dim is omitted")
            self.head_dim = self.dim // self.num_heads
        else:
            self.head_dim = _positive_int(head_dim, "head_dim")
        self.num_kv_heads = (
            self.num_heads if num_kv_heads is None else _positive_int(num_kv_heads, "num_kv_heads")
        )
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.context_dim = self.dim if context_dim is None else _positive_int(
            context_dim, "context_dim"
        )
        for name, value in (
            ("qkv_bias", qkv_bias),
            ("output_bias", output_bias),
            ("qk_norm", qk_norm),
            ("is_causal", is_causal),
            ("use_native_gqa", use_native_gqa),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        self.attention_dropout = _probability(attention_dropout, "attention_dropout")
        self.projection_dropout_probability = _probability(
            projection_dropout, "projection_dropout"
        )
        self.is_causal = is_causal
        self.use_native_gqa = use_native_gqa
        if softmax_scale is None:
            self.softmax_scale = None
        else:
            scale = float(softmax_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("softmax_scale must be finite and positive")
            self.softmax_scale = scale

        self.inner_dim = self.num_heads * self.head_dim
        self.kv_inner_dim = self.num_kv_heads * self.head_dim
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
        self.query_projection = BitLinear(
            self.dim,
            self.inner_dim,
            bias=qkv_bias,
            **linear_kwargs,
        )
        self.key_projection = BitLinear(
            self.context_dim,
            self.kv_inner_dim,
            bias=qkv_bias,
            **linear_kwargs,
        )
        self.value_projection = BitLinear(
            self.context_dim,
            self.kv_inner_dim,
            bias=qkv_bias,
            **linear_kwargs,
        )
        self.output_projection = BitLinear(
            self.inner_dim,
            self.dim,
            bias=output_bias,
            **linear_kwargs,
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, device=device, dtype=dtype)
            if qk_norm
            else nn.Identity()
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=qk_norm_eps, device=device, dtype=dtype)
            if qk_norm
            else nn.Identity()
        )
        self.projection_dropout = nn.Dropout(self.projection_dropout_probability)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
            projection.clear_packed_cache()
        if isinstance(self.query_norm, RMSNorm) and self.query_norm.weight is not None:
            nn.init.ones_(self.query_norm.weight)
        if isinstance(self.key_norm, RMSNorm) and self.key_norm.weight is not None:
            nn.init.ones_(self.key_norm.weight)

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        # projected: [B,Lq,Hq*Dh]; query: [B,Hq,Lq,Dh].
        projected = self.query_projection(hidden_states)
        query = projected.view(batch, query_length, self.num_heads, self.head_dim)
        return self.query_norm(query.transpose(1, 2))

    def _project_key_value(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, key_length, _ = context.shape
        # projected_key/value: [B,Lk,Hkv*Dh].
        projected_key = self.key_projection(context)
        projected_value = self.value_projection(context)
        # key/value: [B,Hkv,Lk,Dh].
        key = projected_key.view(batch, key_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = projected_value.view(
            batch,
            key_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        return self.key_norm(key), value

    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        average_attn_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape[1] != query.shape[1]:
            repeat_factor = query.shape[1] // key.shape[1]
            # key/value: [B,Hq,Lk,Dh] after materializing GQA groups.
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        scale = self.softmax_scale or (1.0 / math.sqrt(self.head_dim))
        # scores: [B,Hq,Lq,Lk], accumulated in float32 for stable softmax.
        scores = torch.matmul(
            query.to(torch.float32),
            key.to(torch.float32).transpose(-2, -1),
        ) * scale
        if mask is not None:
            if mask.dtype is torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.to(torch.float32)
        # probabilities: [B,Hq,Lq,Lk]. All-masked rows become deterministic zero.
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        dropped_probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        ).to(value.dtype)
        # attended: [B,Hq,Lq,Dh].
        attended = torch.matmul(dropped_probabilities, value)
        weights = probabilities.mean(dim=1) if average_attn_weights else probabilities
        return attended, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: KVCache | tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        static_kv: bool = False,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        is_causal: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache] | tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, KVCache
    ]:
        for name, value in (
            ("use_cache", use_cache),
            ("static_kv", static_kv),
            ("need_weights", need_weights),
            ("average_attn_weights", average_attn_weights),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states must have shape [B,Lq,{self.dim}]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0:
            raise ValueError("hidden_states batch and sequence extents must be positive")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        if is_causal is not None and not isinstance(is_causal, bool):
            raise TypeError("is_causal must be bool or None")
        causal = self.is_causal if is_causal is None else is_causal

        if past_key_value is None:
            previous: KVCache | None = None
        elif isinstance(past_key_value, KVCache):
            previous = past_key_value
        elif isinstance(past_key_value, tuple):
            if len(past_key_value) != 2:
                raise ValueError("past_key_value tuple must contain exactly (key, value)")
            previous = KVCache(past_key_value[0], past_key_value[1])
        else:
            raise TypeError("past_key_value must be KVCache, a (key, value) tuple, or None")
        if previous is not None:
            if previous.batch_size != hidden_states.shape[0]:
                raise ValueError("past_key_value batch size does not match hidden_states")
            if previous.num_heads != self.num_kv_heads or previous.head_dim != self.head_dim:
                raise ValueError("past_key_value head geometry does not match this attention module")
            if previous.key.device != hidden_states.device:
                raise ValueError("past_key_value and hidden_states must reside on the same device")

        static_reuse = static_kv and previous is not None
        if static_kv and previous is None and context is None:
            raise ValueError("context is required when initializing a static KV cache")
        if static_reuse and context is not None:
            raise ValueError("context must be omitted when reusing a static KV cache")

        if not static_reuse:
            selected_context = hidden_states if context is None else context
            if not isinstance(selected_context, torch.Tensor):
                raise TypeError("context must be a torch.Tensor or None")
            if selected_context.ndim != 3 or selected_context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"context must have shape [B,Lk,{self.context_dim}]; "
                    f"got {tuple(selected_context.shape)}"
                )
            if selected_context.shape[0] != hidden_states.shape[0] or selected_context.shape[1] <= 0:
                raise ValueError("context batch must match and its sequence extent must be positive")
            if selected_context.device != hidden_states.device:
                raise ValueError("hidden_states and context must reside on the same device")
            if selected_context.dtype != hidden_states.dtype:
                raise TypeError("hidden_states and context must use the same dtype")

        # query: [B,Hq,Lq,Dh].
        query = self._project_query(hidden_states)
        if previous is not None and previous.key.dtype != query.dtype:
            raise TypeError("past_key_value dtype must match the projected query dtype")

        if static_reuse:
            if key_rotary is not None:
                raise ValueError("key_rotary must be omitted when reusing a static KV cache")
            if rotary is not None:
                # query: [B,Hq,Lq,Dh]; cached keys are already rotated.
                query = apply_rotary_embedding(query, rotary)
            cache = previous
        else:
            # new_key/new_value: [B,Hkv,Lnew,Dh].
            new_key, new_value = self._project_key_value(selected_context)
            if rotary is not None:
                selected_key_rotary = rotary if key_rotary is None else key_rotary
                query, new_key = apply_rotary_qk(
                    query,
                    new_key,
                    rotary,
                    selected_key_rotary,
                )
            elif key_rotary is not None:
                raise ValueError("key_rotary requires rotary query frequencies")

            if previous is None:
                cache = KVCache(new_key, new_value)
            else:
                cache = previous.append(new_key, new_value)
        key, value = cache.key, cache.value
        batch_size, _, query_length, _ = query.shape
        key_length = key.shape[-2]

        # combined_mask: broadcastable to [B,Hq,Lq,Lk].
        combined_mask = _canonical_attention_mask(
            attention_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            query_length=query_length,
            key_length=key_length,
            device=query.device,
            dtype=query.dtype,
        )
        combined_mask = _merge_masks(
            combined_mask,
            _padding_mask(
                key_padding_mask,
                batch_size=batch_size,
                key_length=key_length,
                device=query.device,
                dtype=query.dtype,
            ),
            dtype=query.dtype,
        )
        fast_causal = (
            causal
            and combined_mask is None
            and past_key_value is None
            and context is None
            and query_length == key_length
            and not need_weights
        )
        if causal and not fast_causal:
            combined_mask = _merge_masks(
                combined_mask,
                _causal_mask(query_length, key_length, device=query.device),
                dtype=query.dtype,
            )

        weights: torch.Tensor | None = None
        if need_weights:
            attended, weights = self._manual_attention(
                query,
                key,
                value,
                combined_mask,
                average_attn_weights=average_attn_weights,
            )
        else:
            enable_gqa = self.use_native_gqa and self.num_heads != self.num_kv_heads
            if self.num_heads != self.num_kv_heads and not enable_gqa:
                repeat_factor = self.num_heads // self.num_kv_heads
                # key/value: [B,Hq,Lk,Dh].
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)
            # attended: [B,Hq,Lq,Dh]. SDPA selects Flash/efficient/math kernels.
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=combined_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=fast_causal,
                scale=self.softmax_scale,
                enable_gqa=enable_gqa,
            )

        # merged: [B,Lq,Hq*Dh]; output: [B,Lq,dim].
        merged = attended.transpose(1, 2).reshape(batch_size, query_length, self.inner_dim)
        output = self.projection_dropout(self.output_projection(merged))
        if use_cache and need_weights:
            assert weights is not None
            return output, weights, cache
        if use_cache:
            return output, cache
        if need_weights:
            assert weights is not None
            return output, weights
        return output

    @torch.no_grad()
    def pack_weights(
        self,
        layout: int | WeightLayout | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> dict[str, PackedTernaryWeight]:
        """Pack all four projection matrices for inference."""

        return {
            "query": self.query_projection.pack_weights(layout, backend=backend),
            "key": self.key_projection.pack_weights(layout, backend=backend),
            "value": self.value_projection.pack_weights(layout, backend=backend),
            "output": self.output_projection.pack_weights(layout, backend=backend),
        }

    def clear_packed_cache(self) -> None:
        self.query_projection.clear_packed_cache()
        self.key_projection.clear_packed_cache()
        self.value_projection.clear_packed_cache()
        self.output_projection.clear_packed_cache()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, context_dim={self.context_dim}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"attention_dropout={self.attention_dropout:g}, "
            f"projection_dropout={self.projection_dropout_probability:g}, "
            f"is_causal={self.is_causal}, use_native_gqa={self.use_native_gqa}"
        )


MultiheadAttention = Attention
