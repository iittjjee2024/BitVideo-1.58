"""Lazy backend discovery for BitVideo operators.

Optional native and Triton modules are never imported while importing
``bitvideo`` itself. Discovery is cached, thread-safe, inspectable, and can be
refreshed after an in-place extension build.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import torch

from .types import Backend, normalize_backend

_REQUIRED_EXTENSION_SYMBOLS = (
    "pack_ternary_weights",
    "pack_ternary_int8",
    "unpack_ternary_weights",
    "convert_ternary_layout",
    "quantize_activations",
    "bit_linear_forward",
)
_UNSET = object()
_LOCK = threading.RLock()
_extension: object | ModuleType | None = _UNSET
_extension_error: str | None = None
_triton_forward: object | Callable[..., torch.Tensor] | None = _UNSET
_triton_error: str | None = None


@dataclass(frozen=True)
class BackendStatus:
    """Availability snapshot for all accelerated execution tiers."""

    cuda_extension: bool
    triton: bool
    torch_int_mm: bool
    cuda_extension_error: str | None = None
    triton_error: str | None = None

    @property
    def available(self) -> tuple[Backend, ...]:
        result: list[Backend] = []
        if self.cuda_extension:
            result.append(Backend.CUDA_EXTENSION)
        if self.triton:
            result.append(Backend.TRITON)
        result.append(Backend.TORCH)
        return tuple(result)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _format_import_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def get_extension(*, required: bool = False) -> ModuleType | None:
    """Return ``bitvideo._C`` when loadable, otherwise ``None``.

    Set ``BITVIDEO_DISABLE_CUDA_EXTENSION=1`` to force the Python tiers.
    ``required=True`` converts the cached diagnostic into an actionable error.
    """

    global _extension, _extension_error
    with _LOCK:
        if _extension is _UNSET:
            if _env_flag("BITVIDEO_DISABLE_CUDA_EXTENSION"):
                _extension = None
                _extension_error = "disabled by BITVIDEO_DISABLE_CUDA_EXTENSION"
            else:
                try:
                    module = importlib.import_module("bitvideo._C")
                    missing = [name for name in _REQUIRED_EXTENSION_SYMBOLS if not hasattr(module, name)]
                    if missing:
                        raise ImportError(
                            "bitvideo._C is missing required symbols: " + ", ".join(missing)
                        )
                    _extension = module
                    _extension_error = None
                except (ImportError, OSError, RuntimeError) as exc:
                    _extension = None
                    _extension_error = _format_import_error(exc)
        module = _extension if isinstance(_extension, ModuleType) else None
        if required and module is None:
            detail = _extension_error or "unknown extension load failure"
            raise RuntimeError(
                "the BitVideo CUDA extension was requested but is unavailable: " + detail
            )
        return module


def register_triton_backend(forward: Callable[..., torch.Tensor] | None) -> None:
    """Register an in-process Triton forward callable.

    The callable must implement the extension-compatible keyword contract used
    by :func:`bitvideo.ops.bit_linear`. Passing ``None`` clears the registration
    and re-enables lazy module discovery.
    """

    global _triton_forward, _triton_error
    if forward is not None and not callable(forward):
        raise TypeError("forward must be callable or None")
    with _LOCK:
        _triton_forward = _UNSET if forward is None else forward
        _triton_error = None


def get_triton_forward(*, required: bool = False) -> Callable[..., torch.Tensor] | None:
    """Return the optional Triton BitLinear callable when available."""

    global _triton_forward, _triton_error
    with _LOCK:
        if _triton_forward is _UNSET:
            if _env_flag("BITVIDEO_DISABLE_TRITON"):
                _triton_forward = None
                _triton_error = "disabled by BITVIDEO_DISABLE_TRITON"
            elif importlib.util.find_spec("triton") is None:
                _triton_forward = None
                _triton_error = "the optional 'triton' package is not installed"
            else:
                try:
                    module = importlib.import_module("bitvideo.triton.bitlinear")
                    forward = getattr(module, "bit_linear_forward")
                    if not callable(forward):
                        raise TypeError("bitvideo.triton.bitlinear.bit_linear_forward is not callable")
                    _triton_forward = forward
                    _triton_error = None
                except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
                    _triton_forward = None
                    _triton_error = _format_import_error(exc)
        forward = _triton_forward if callable(_triton_forward) else None
        if required and forward is None:
            detail = _triton_error or "unknown Triton backend load failure"
            raise RuntimeError("the BitVideo Triton backend was requested but is unavailable: " + detail)
        return forward


def backend_status(*, refresh: bool = False) -> BackendStatus:
    """Return backend availability and cached import diagnostics."""

    if refresh:
        refresh_backends()
    extension = get_extension()
    triton_forward = get_triton_forward()
    with _LOCK:
        return BackendStatus(
            cuda_extension=extension is not None,
            triton=triton_forward is not None,
            torch_int_mm=callable(getattr(torch, "_int_mm", None)),
            cuda_extension_error=_extension_error,
            triton_error=_triton_error,
        )


def refresh_backends() -> BackendStatus:
    """Clear discovery caches and probe optional backends again."""

    global _extension, _extension_error, _triton_forward, _triton_error
    with _LOCK:
        _extension = _UNSET
        _extension_error = None
        _triton_forward = _UNSET
        _triton_error = None
    return backend_status(refresh=False)


def requested_backend(value: str | Backend | None) -> Backend:
    """Resolve an explicit backend or the process-wide ``BITVIDEO_BACKEND`` override."""

    normalized = normalize_backend(value)
    if normalized is Backend.AUTO:
        environment = os.environ.get("BITVIDEO_BACKEND")
        if environment:
            return normalize_backend(environment)
    return normalized


def is_compiling() -> bool:
    """Return whether TorchDynamo/``torch.compile`` is currently tracing."""

    compiler: Any = getattr(torch, "compiler", None)
    probe = getattr(compiler, "is_compiling", None)
    return bool(probe()) if callable(probe) else False
