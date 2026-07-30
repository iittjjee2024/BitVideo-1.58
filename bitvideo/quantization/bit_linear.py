"""Quantization-aware and packed-inference BitLinear module."""

from __future__ import annotations

import math
import struct
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitvideo.ops import (
    ActivationGranularity,
    Backend,
    KernelVariant,
    PackedTernaryWeight,
    WeightLayout,
    bit_linear as packed_bit_linear,
    pack_ternary_weight,
)
from bitvideo.ops.types import normalize_backend, normalize_layout, normalize_variant

from .quantization import ActivationQuantizer, QuantizationConfig, TernaryWeightQuantizer

_PACKED_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
# Native GEMV kernels accept row-major weights only, so an explicit request for
# one of these variants must never be converted to another layout.
_GEMV_ONLY_VARIANTS = frozenset(
    {
        KernelVariant.GEMV_1WARP,
        KernelVariant.GEMV_SPLIT_K,
        KernelVariant.GEMV_WIDE,
    }
)
_NATIVE_ACTIVATION_EPS = 1.0e-5
_LAYOUT_ALIASES = {
    "row": WeightLayout.ROW_MAJOR,
    "row_major": WeightLayout.ROW_MAJOR,
    "column": WeightLayout.COLUMN_MAJOR,
    "column_major": WeightLayout.COLUMN_MAJOR,
    "blocked": WeightLayout.BLOCKED_N64,
    "blocked_n64": WeightLayout.BLOCKED_N64,
    "mma": WeightLayout.MMA_INTERLEAVED,
    "mma_interleaved": WeightLayout.MMA_INTERLEAVED,
}


def _same_float32(left: float, right: float) -> bool:
    """Return whether two validated scalars have identical float32 encodings."""

    return struct.pack("<f", float(left)) == struct.pack("<f", float(right))


def _is_compiling() -> bool:
    return bool(torch.compiler.is_compiling())


def _autocast_enabled(device_type: str) -> bool:
    """Return autocast state for the input device on supported PyTorch versions."""

    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return bool(torch.is_autocast_enabled())


@torch.library.custom_op("bitvideo::_packed_weight_cache_matches", mutates_args=())
def _packed_weight_cache_matches(
    weight: torch.Tensor,
    expected_version: int,
    expected_data_ptr: int,
    expected_stride_0: int,
    expected_stride_1: int,
    expected_storage_offset: int,
) -> torch.Tensor:
    """Return a scalar runtime token for an opaque compiled-cache guard."""

    try:
        current_version = weight._version
    except RuntimeError:
        current_version = -1
    current_data_ptr = weight.data_ptr() if weight.device.type != "meta" else -1
    is_current = (
        current_version == expected_version
        and current_data_ptr == expected_data_ptr
        and weight.stride(0) == expected_stride_0
        and weight.stride(1) == expected_stride_1
        and weight.storage_offset() == expected_storage_offset
    )
    if not is_current:
        raise RuntimeError(
            "BitLinear packed cache is stale; repack weights and recompile the graph"
        )
    # result: [] bool on the weight device. The consumed token prevents DCE.
    return torch.ones((), dtype=torch.bool, device=weight.device)


@_packed_weight_cache_matches.register_fake
def _packed_weight_cache_matches_fake(
    weight: torch.Tensor,
    expected_version: int,
    expected_data_ptr: int,
    expected_stride_0: int,
    expected_stride_1: int,
    expected_storage_offset: int,
) -> torch.Tensor:
    del (
        expected_version,
        expected_data_ptr,
        expected_stride_0,
        expected_stride_1,
        expected_storage_offset,
    )
    # result: [] fake bool on the weight device.
    return torch.empty((), dtype=torch.bool, device=weight.device)


class BitLinear(nn.Module):
    """Linear projection with QAT training and packed W1.58A8 inference.

    Training (and any gradient-enabled evaluation) uses differentiable fake
    quantization followed by ``torch.nn.functional.linear``. Evaluation under
    ``torch.no_grad``/``torch.inference_mode`` lazily packs the current weight
    and invokes the three-tier operator dispatcher.
    """

    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
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
        if (
            isinstance(in_features, bool)
            or isinstance(out_features, bool)
            or not isinstance(in_features, int)
            or not isinstance(out_features, int)
            or in_features <= 0
            or out_features <= 0
        ):
            raise ValueError("in_features and out_features must be positive integers")
        if not isinstance(bias, bool):
            raise TypeError("bias must be bool")
        if isinstance(split_k, bool) or not isinstance(split_k, int) or not 0 <= split_k <= 64:
            raise ValueError("split_k must be an integer in [0, 64]")
        for name, value in (
            ("autotune", autotune),
            ("use_packed_inference", use_packed_inference),
            ("auto_pack", auto_pack),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if dtype is not None:
            if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                raise TypeError("BitLinear parameters must use a floating-point dtype")
        if quantization is not None and not isinstance(quantization, QuantizationConfig):
            raise TypeError("quantization must be a QuantizationConfig or None")

        self.in_features = in_features
        self.out_features = out_features
        self.quantization_config = quantization or QuantizationConfig()
        weight_config = self.quantization_config.weight
        activation_config = self.quantization_config.activation
        self._packed_quantization_config_compatible = (
            weight_config.enabled
            and activation_config.enabled
            and _same_float32(weight_config.threshold_factor, 0.5)
            and activation_config.bits == 8
            and _same_float32(activation_config.eps, _NATIVE_ACTIVATION_EPS)
            and activation_config.granularity
            in {ActivationGranularity.PER_TENSOR, ActivationGranularity.PER_TOKEN}
        )
        self.weight_quantizer = TernaryWeightQuantizer(weight_config)
        self.activation_quantizer = ActivationQuantizer(activation_config)
        self.inference_layout = self._normalize_inference_layout(inference_layout)
        self.backend = normalize_backend(backend)
        self.variant = normalize_variant(variant)
        self.split_k = split_k
        self.autotune = autotune
        self.use_packed_inference = use_packed_inference
        self.auto_pack = auto_pack

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        # weight: [N, K]; bias parameter: [N] when enabled.
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Derived cache buffers are intentionally non-persistent: checkpoints
        # store the master weight and regenerate backend/layout-specific bytes.
        self.register_buffer(
            "_packed_data",
            torch.empty(0, dtype=torch.int32, device=self.weight.device),
            persistent=False,
        )
        self.register_buffer(
            "_packed_scale",
            torch.empty(0, dtype=torch.float32, device=self.weight.device),
            persistent=False,
        )
        self._packed_n_padded = 0
        self._packed_k_padded = 0
        self._packed_layout = WeightLayout.ROW_MAJOR
        self._packed_scale_mode = self.quantization_config.weight.scale_mode
        self._packed_weight_version: int | None = None
        self._packed_weight_parameter_id = 0
        self._packed_weight_data_ptr: int | None = None
        self._packed_weight_stride: tuple[int, int] | None = None
        self._packed_weight_storage_offset: int | None = None
        self._packed_object_cache: PackedTernaryWeight | None = None

    @staticmethod
    def _normalize_inference_layout(value: str | int | WeightLayout) -> str | WeightLayout:
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_")
            if key == "auto":
                return "auto"
            if key in _LAYOUT_ALIASES:
                return _LAYOUT_ALIASES[key]
            legal = "auto, row_major, column_major, blocked_n64, mma_interleaved"
            raise ValueError(f"invalid inference_layout {value!r}; expected one of {legal}")
        return normalize_layout(value)

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        try:
            return tensor._version
        except RuntimeError:
            # Parameters created inside inference_mode do not expose a version counter.
            return None

    @staticmethod
    def _tensor_data_ptr(tensor: torch.Tensor) -> int | None:
        if tensor.device.type == "meta":
            return None
        return tensor.data_ptr()

    def _validate_master_parameters(self) -> None:
        if self.weight.ndim != 2 or tuple(self.weight.shape) != (
            self.out_features,
            self.in_features,
        ):
            raise RuntimeError(
                "master weight shape changed: expected "
                f"[{self.out_features}, {self.in_features}], got {tuple(self.weight.shape)}"
            )
        if not self.weight.is_floating_point():
            raise RuntimeError("master weight must remain floating point")
        if self.bias is not None:
            if self.bias.ndim != 1 or self.bias.numel() != self.out_features:
                raise RuntimeError(
                    f"bias must have shape [{self.out_features}]; got {tuple(self.bias.shape)}"
                )
            if not self.bias.is_floating_point():
                raise RuntimeError("bias must remain floating point")
            if self.bias.device != self.weight.device:
                raise RuntimeError("weight and bias must remain on the same device")
            if self.bias.dtype != self.weight.dtype:
                raise RuntimeError("weight and bias must use the same dtype")

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, "_packed_data"):
            self.clear_packed_cache()

    def _apply(self, fn: Any, recurse: bool = True) -> "BitLinear":
        # Applying an arbitrary module transform can change master-weight
        # device or precision. Drop derived bytes instead of risking stale scales.
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_packed_data"):
            self.clear_packed_cache()
        return result

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Loading can copy into or replace a Parameter; either invalidates bytes.
        if hasattr(self, "_packed_data"):
            self.clear_packed_cache()

    def train(self, mode: bool = True) -> "BitLinear":
        super().train(mode)
        if mode and hasattr(self, "_packed_data"):
            self.clear_packed_cache()
        return self

    def clear_packed_cache(self) -> None:
        """Discard all derived packed storage without modifying parameters."""

        device = self.weight.device
        # packed_data: [0] int32; packed_scale: [0] float32.
        self._packed_data = torch.empty(0, dtype=torch.int32, device=device)
        self._packed_scale = torch.empty(0, dtype=torch.float32, device=device)
        self._packed_n_padded = 0
        self._packed_k_padded = 0
        self._packed_layout = WeightLayout.ROW_MAJOR
        self._packed_scale_mode = self.quantization_config.weight.scale_mode
        self._packed_weight_version = None
        self._packed_weight_parameter_id = 0
        self._packed_weight_data_ptr = None
        self._packed_weight_stride = None
        self._packed_weight_storage_offset = None
        self._packed_object_cache = None

    def _packed_compatible(self) -> bool:
        return (
            self._packed_quantization_config_compatible
            and self.weight.dtype in _PACKED_FLOAT_DTYPES
        )

    def _layout_for(self, x: torch.Tensor) -> WeightLayout:
        if self.inference_layout != "auto":
            return self.inference_layout
        m = x.numel() // self.in_features
        if x.is_cuda and m > 8:
            return WeightLayout.MMA_INTERLEAVED
        return WeightLayout.ROW_MAJOR

    def _auto_layout_for_cache(
        self,
        x: torch.Tensor,
        cached_layout: WeightLayout,
    ) -> WeightLayout:
        """Return the layout to use when an ``auto`` packed cache already exists.

        The rule is deliberately narrow: keep the cached layout and perform
        exactly one upgrade, from row-major to MMA-interleaved, when a CUDA
        problem grows past the native GEMV bound. Row-major native kernels are
        GEMV-only, so without that upgrade an explicit native backend would stay
        permanently ineligible and ``auto`` would silently serve the portable
        tier for the rest of the module's life.

        Everything else is preserved rather than "optimized":

        * MMA-interleaved caches are never downgraded at small ``M``, so an
          offline prepack survives short sequences.
        * Blocked-N64 and column-major caches are kept as the deliberate DP4A
          and ablation layouts they are. Column-major has no native kernel, so
          an explicit native backend will reject it at any ``M``; that refusal
          is reported rather than silently rewritten.
        * A requested GEMV variant pins row-major, because those kernels accept
          no other layout. Upgrading would convert a working configuration into
          a permanently ineligible one.
        * CPU has no native tier and the portable tier decodes every layout, so
          no CPU repack can help.

        Because the only transition is row-major to MMA-interleaved and MMA has
        no exit, repeated crossings of the bound settle after a single repack
        instead of thrashing.
        """

        if not x.is_cuda:
            return cached_layout
        if self.variant in _GEMV_ONLY_VARIANTS:
            return cached_layout
        if cached_layout is WeightLayout.ROW_MAJOR:
            m = x.numel() // self.in_features
            if m > 8:
                return WeightLayout.MMA_INTERLEAVED
        return cached_layout

    def _unavailable_backend_message(
        self,
        desired_layout: WeightLayout | None,
        *,
        compiling: bool,
    ) -> str:
        """Explain why an explicitly requested native backend cannot execute."""

        if compiling:
            # Prepacking cannot rescue this case: the dispatcher refuses native
            # backends for the whole capture, so naming a layout would mislead.
            return (
                f"explicit {self.backend.value} inference is unavailable inside torch.compile "
                "because native backends are bypassed while tracing; run this module eagerly, "
                "or select the auto/torch backend for compiled execution"
            )
        requirement = (
            "prepacked weights"
            if desired_layout is None
            else f"weights prepacked in the {desired_layout.name} layout"
        )
        return (
            f"explicit {self.backend.value} packed inference requires {requirement}; "
            "call pack_weights() before inference or enable auto_pack"
        )

    def _cache_is_current(self, layout: WeightLayout | None = None) -> bool:
        if self._packed_data.numel() == 0:
            return False
        # Inference tensors have no version counter, so their derived bytes are
        # deliberately single-use rather than silently reusable.
        if self._packed_weight_version is None:
            return False
        if not _is_compiling():
            if self._packed_weight_parameter_id != id(self.weight):
                return False
            if self._packed_weight_version != self._tensor_version(self.weight):
                return False
            # data_ptr catches direct ``parameter.data = other_tensor`` replacement.
            if self._packed_weight_data_ptr != self._tensor_data_ptr(self.weight):
                return False
            if self._packed_weight_stride != tuple(self.weight.stride()):
                return False
            if self._packed_weight_storage_offset != self.weight.storage_offset():
                return False
        if self._packed_data.device != self.weight.device:
            return False
        if self._packed_scale.device != self.weight.device:
            return False
        if self._packed_scale_mode is not self.quantization_config.weight.scale_mode:
            return False
        return layout is None or self._packed_layout is layout

    def _assert_compiled_cache_current(self) -> None:
        """Fail a captured graph when its master Parameter changed after packing."""

        if (
            self._packed_weight_version is None
            or self._packed_weight_data_ptr is None
            or self._packed_weight_stride is None
            or self._packed_weight_storage_offset is None
        ):
            raise RuntimeError(
                "compiled packed inference requires a versioned, prepacked master weight"
            )
        # cache_matches: [] bool. The custom op is opaque to Dynamo and checks
        # the live version, storage identity, offset, and logical strides.
        cache_matches = _packed_weight_cache_matches(
            self.weight,
            self._packed_weight_version,
            self._packed_weight_data_ptr,
            self._packed_weight_stride[0],
            self._packed_weight_stride[1],
            self._packed_weight_storage_offset,
        )
        torch._check_tensor_all(
            cache_matches,
            lambda: "BitLinear packed cache is stale; repack weights and recompile the graph",
        )

    def _cached_packed(self, layout: WeightLayout | None = None) -> PackedTernaryWeight | None:
        if not self._cache_is_current(layout):
            return None
        if self._packed_object_cache is None:
            self._packed_object_cache = PackedTernaryWeight(
                self._packed_data,
                self._packed_scale,
                self.out_features,
                self.in_features,
                self._packed_n_padded,
                self._packed_k_padded,
                self._packed_layout,
                self._packed_scale_mode,
            )
        return self._packed_object_cache

    @torch.no_grad()
    def pack_weights(
        self,
        layout: int | WeightLayout | None = None,
        *,
        backend: str | Backend | None = None,
    ) -> PackedTernaryWeight:
        """Pack the current master weight and refresh the inference cache."""

        self._validate_master_parameters()
        if not self._packed_compatible():
            raise RuntimeError(
                "packed inference requires float16/bfloat16/float32 master weights, enabled "
                "W1.58/A8 quantization, threshold_factor=0.5, activation eps=1e-5, "
                "8-bit activations, and tensor/token activation scales"
            )
        if self.weight.device.type == "meta":
            raise RuntimeError("cannot pack a meta-device weight")
        if layout is None:
            if self.inference_layout == "auto":
                selected_layout = (
                    WeightLayout.MMA_INTERLEAVED
                    if self.weight.is_cuda
                    else WeightLayout.ROW_MAJOR
                )
            else:
                selected_layout = self.inference_layout
        else:
            selected_layout = normalize_layout(layout)

        runtime_backend = self.backend if backend is None else normalize_backend(backend)
        # Triton consumes the common packed format but does not provide an
        # offline packer. Use the portable codec and keep Triton for execution.
        packing_backend = Backend.TORCH if runtime_backend is Backend.TRITON else runtime_backend
        packed = pack_ternary_weight(
            self.weight.detach(),
            layout=selected_layout,
            scale_mode=self.quantization_config.weight.scale_mode,
            eps=self.quantization_config.weight.eps,
            backend=packing_backend,
        )
        self._packed_data = packed.data
        self._packed_scale = packed.scale
        self._packed_n_padded = packed.n_padded
        self._packed_k_padded = packed.k_padded
        self._packed_layout = packed.layout
        self._packed_scale_mode = packed.scale_mode
        self._packed_weight_version = self._tensor_version(self.weight)
        self._packed_weight_parameter_id = id(self.weight)
        self._packed_weight_data_ptr = self._tensor_data_ptr(self.weight)
        self._packed_weight_stride = tuple(self.weight.stride())
        self._packed_weight_storage_offset = self.weight.storage_offset()
        self._packed_object_cache = packed
        return packed

    @property
    def packed_weight(self) -> PackedTernaryWeight | None:
        """Return the current packed cache, or ``None`` when absent/stale."""

        return self._cached_packed()

    def _qat_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_quantized: [..., K]; weight_quantized: [N, K].
        x_quantized = self.activation_quantizer(x)
        weight_quantized = self.weight_quantizer(self.weight)
        # output: [..., N].
        return F.linear(x_quantized, weight_quantized, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_master_parameters()
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"x must have shape [..., {self.in_features}]; got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point; got {x.dtype}")
        if x.device != self.weight.device:
            raise ValueError(f"x must be on {self.weight.device}; got {x.device}")
        autocast_enabled = _autocast_enabled(x.device.type)
        if x.dtype != self.weight.dtype and not autocast_enabled:
            raise TypeError(
                "x and BitLinear parameters must use the same dtype outside autocast; "
                f"got x={x.dtype}, weight={self.weight.dtype}"
            )

        gradient_required = torch.is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        )
        if (
            self.training
            or gradient_required
            or autocast_enabled
            or not self.use_packed_inference
            or x.dtype not in _PACKED_FLOAT_DTYPES
            or not self._packed_compatible()
        ):
            return self._qat_forward(x)

        compiling = _is_compiling()
        # desired_layout stays unresolved until it is actually needed, so a
        # compiled graph without a usable cache never specializes on the
        # shape-driven heuristic it would immediately discard.
        desired_layout: WeightLayout | None = None
        if self.inference_layout == "auto":
            packed = self._cached_packed()
            if packed is not None:
                if compiling:
                    # Repacking is impossible mid-capture, and native backends are
                    # bypassed while tracing, so every layout runs the portable
                    # tier here. Keeping the cache lets one graph span shapes that
                    # straddle the eager GEMV bound.
                    desired_layout = self._packed_layout
                else:
                    upgraded_layout = self._auto_layout_for_cache(x, self._packed_layout)
                    if upgraded_layout is not self._packed_layout and self.auto_pack:
                        # Discard the cache only when a repack can actually run.
                        desired_layout = upgraded_layout
                        packed = None
                    else:
                        # A better layout exists but may not be produced here, so
                        # keep serving packed inference from the cache. Falling
                        # back to QAT would silently change numerics while a
                        # usable prepack sits unused.
                        desired_layout = self._packed_layout
        else:
            desired_layout = self.inference_layout
            packed = self._cached_packed(desired_layout)
        if packed is None:
            # Cache mutation cannot safely occur inside a full-graph capture.
            # Portable compiled execution uses QAT unless the module was prepacked.
            if not self.auto_pack or compiling:
                if self.backend in {Backend.CUDA_EXTENSION, Backend.TRITON}:
                    raise RuntimeError(
                        self._unavailable_backend_message(desired_layout, compiling=compiling)
                    )
                return self._qat_forward(x)
            if desired_layout is None:
                desired_layout = self._layout_for(x)
            packed = self.pack_weights(desired_layout)

        if compiling:
            self._assert_compiled_cache_current()
        return packed_bit_linear(
            x,
            packed,
            self.bias,
            out_dtype=x.dtype,
            activation_granularity=self.quantization_config.activation.granularity,
            backend=self.backend,
            variant=self.variant,
            split_k=self.split_k,
            autotune=self.autotune,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        quantization: QuantizationConfig | None = None,
        inference_layout: str | int | WeightLayout = "auto",
        backend: str | Backend = Backend.AUTO,
        **kwargs: Any,
    ) -> "BitLinear":
        """Create a BitLinear with an exact copy of an ``nn.Linear`` state."""

        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be torch.nn.Linear")
        result = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quantization=quantization,
            inference_layout=inference_layout,
            backend=backend,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **kwargs,
        )
        with torch.no_grad():
            # result.weight/source weight: [N, K].
            result.weight.copy_(linear.weight)
            if linear.bias is not None and result.bias is not None:
                # result.bias/source bias: [N].
                result.bias.copy_(linear.bias)
        result.weight.requires_grad_(linear.weight.requires_grad)
        if result.bias is not None and linear.bias is not None:
            result.bias.requires_grad_(linear.bias.requires_grad)
        result.train(linear.training)
        return result

    def to_linear(self) -> nn.Linear:
        """Materialize a standard ``nn.Linear`` containing the master weights."""

        result = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            # result/master weight: [N, K].
            result.weight.copy_(self.weight)
            if self.bias is not None and result.bias is not None:
                # result/master bias: [N].
                result.bias.copy_(self.bias)
        result.weight.requires_grad_(self.weight.requires_grad)
        if result.bias is not None and self.bias is not None:
            result.bias.requires_grad_(self.bias.requires_grad)
        result.train(self.training)
        return result

    def extra_repr(self) -> str:
        layout = (
            self.inference_layout
            if isinstance(self.inference_layout, str)
            else self.inference_layout.name
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_scale={self.quantization_config.weight.scale_mode.name}, "
            f"activation={self.quantization_config.activation.granularity.name}, "
            f"layout={layout}, backend={self.backend.value}, packed={self._cache_is_current()}"
        )
