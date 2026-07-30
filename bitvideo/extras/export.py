"""Model export utilities for deployment (ONNX, TorchScript)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    *,
    example_inputs: tuple[Any, ...] | None = None,
    method: str = "trace",
) -> Path:
    """Export model to TorchScript format.

    Args:
        model: The model to export (should be in eval mode).
        output_path: Path for the exported .pt file.
        example_inputs: Example inputs for tracing (required for trace method).
        method: Export method - 'trace' or 'script'.

    Returns:
        Path to the exported file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if method == "trace":
        if example_inputs is None:
            raise ValueError("example_inputs required for trace export")
        scripted = torch.jit.trace(model, example_inputs)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"method must be 'trace' or 'script'; got {method!r}")

    scripted.save(str(output_path))
    return output_path


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    example_inputs: tuple[Any, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> Path:
    """Export model to ONNX format.

    Args:
        model: The model to export (should be in eval mode).
        output_path: Path for the exported .onnx file.
        example_inputs: Example inputs as a tuple.
        input_names: Names for the input tensors.
        output_names: Names for the output tensors.
        dynamic_axes: Dynamic axis specifications.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(example_inputs))]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    return output_path
