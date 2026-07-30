"""Quantization-aware feed-forward networks for BitVideo transformer blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitvideo.ops import (
    ActivationGranularity,
    Backend,
    KernelVariant,
    PackedTernaryWeight,
    WeightLayout,
)
from bitvideo.quantization import BitLinear, QuantizationConfig

_ACTIVATION_ALIASES = {
    "gelu": "gelu",
    "gelu_erf": "gelu",
    "gelu_tanh": "gelu_tanh",
    "geglu": "gelu",
    "geglu_tanh": "gelu_tanh",
    "silu": "silu",
    "swish": "silu",
    "swiglu": "silu",
    "relu": "relu",
    "relu2": "relu2",
    "squared_relu": "relu2",
}
_GATED_ALIASES = {"geglu", "geglu_tanh", "swiglu"}


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


def _normalize_activation(value: str) -> tuple[str, bool]:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _ACTIVATION_ALIASES:
        legal = ", ".join(sorted(_ACTIVATION_ALIASES))
        raise ValueError(f"activation must be one of {legal}; got {value!r}")
    return _ACTIVATION_ALIASES[normalized], normalized in _GATED_ALIASES


class FeedForward(nn.Module):
    """BitLinear MLP supporting dense, GEGLU, and SwiGLU formulations."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_features: int | None = None,
        expansion_ratio: float = 4.0,
        multiple_of: int = 1,
        activation: str = "swiglu",
        gated: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        output_dropout: float | None = None,
        chunk_size: int | None = None,
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
        multiple = _positive_int(multiple_of, "multiple_of")
        if hidden_features is None:
            if isinstance(expansion_ratio, bool):
                raise TypeError("expansion_ratio must be a real number, not bool")
            ratio = float(expansion_ratio)
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("expansion_ratio must be finite and positive")
            unrounded = math.ceil(self.dim * ratio)
            self.hidden_features = ((unrounded + multiple - 1) // multiple) * multiple
        else:
            self.hidden_features = _positive_int(hidden_features, "hidden_features")
            if self.hidden_features % multiple:
                raise ValueError(
                    f"hidden_features={self.hidden_features} must be divisible by "
                    f"multiple_of={multiple}"
                )
        normalized_activation, implied_gating = _normalize_activation(activation)
        if gated is None:
            self.gated = implied_gating
        elif not isinstance(gated, bool):
            raise TypeError("gated must be bool or None")
        else:
            self.gated = gated
        if implied_gating and not self.gated:
            raise ValueError(f"activation={activation!r} requires gated=True")
        if not isinstance(bias, bool):
            raise TypeError("bias must be bool")
        self.activation_name = normalized_activation
        self.dropout_probability = _probability(dropout, "dropout")
        self.output_dropout_probability = _probability(
            self.dropout_probability if output_dropout is None else output_dropout,
            "output_dropout",
        )
        if chunk_size is not None:
            _positive_int(chunk_size, "chunk_size")
        self.chunk_size = chunk_size

        projection_features = self.hidden_features * (2 if self.gated else 1)
        linear_kwargs = {
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
        self.input_projection = BitLinear(
            self.dim,
            projection_features,
            bias=bias,
            **linear_kwargs,
        )
        self.output_projection = BitLinear(
            self.hidden_features,
            self.dim,
            bias=bias,
            **linear_kwargs,
        )
        self.hidden_dropout = nn.Dropout(self.dropout_probability)
        self.output_dropout = nn.Dropout(self.output_dropout_probability)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        # output: x.shape.
        if self.activation_name == "silu":
            return F.silu(x)
        if self.activation_name == "gelu_tanh":
            return F.gelu(x, approximate="tanh")
        if self.activation_name == "gelu":
            return F.gelu(x, approximate="none")
        if self.activation_name == "relu":
            return F.relu(x)
        # relu2 output: x.shape.
        activated = F.relu(x)
        return activated * activated

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        # projected: [..., L, hidden_features*(2 if gated else 1)].
        projected = self.input_projection(x)
        if self.gated:
            # gate/value: [..., L, hidden_features].
            gate, value = projected.chunk(2, dim=-1)
            hidden = self._activation(gate) * value
        else:
            # hidden: [..., L, hidden_features].
            hidden = self._activation(projected)
        hidden = self.hidden_dropout(hidden)
        # output: [..., L, dim].
        return self.output_dropout(self.output_projection(hidden))

    def forward(
        self,
        x: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [..., L, {self.dim}]; got {tuple(x.shape)}")
        if x.shape[-2] <= 0:
            raise ValueError("x sequence extent must be positive")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point; got {x.dtype}")
        selected_chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if selected_chunk_size is None:
            return self._forward_chunk(x)
        selected_chunk_size = _positive_int(selected_chunk_size, "chunk_size")
        if selected_chunk_size >= x.shape[-2]:
            return self._forward_chunk(x)
        activation_config = self.input_projection.quantization_config.activation
        if (
            activation_config.enabled
            and activation_config.granularity is ActivationGranularity.PER_TENSOR
        ):
            raise ValueError(
                "chunked feed-forward execution is incompatible with enabled per-tensor "
                "activation quantization because each chunk would derive a different scale"
            )
        # chunks: tuple of [..., Li, dim], partitioned along sequence.
        chunks = torch.split(x, selected_chunk_size, dim=-2)
        # outputs: list of [..., Li, dim]; output: x.shape.
        outputs = [self._forward_chunk(chunk) for chunk in chunks]
        return torch.cat(outputs, dim=-2)

    @torch.no_grad()
    def pack_weights(
        self,
        layout: int | WeightLayout | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> tuple[PackedTernaryWeight, PackedTernaryWeight]:
        """Pack input and output projections for inference."""

        return (
            self.input_projection.pack_weights(layout, backend=backend),
            self.output_projection.pack_weights(layout, backend=backend),
        )

    def clear_packed_cache(self) -> None:
        self.input_projection.clear_packed_cache()
        self.output_projection.clear_packed_cache()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_features={self.hidden_features}, "
            f"activation={self.activation_name!r}, gated={self.gated}, "
            f"dropout={self.dropout_probability:g}, "
            f"output_dropout={self.output_dropout_probability:g}, "
            f"chunk_size={self.chunk_size}"
        )


BitFeedForward = FeedForward
