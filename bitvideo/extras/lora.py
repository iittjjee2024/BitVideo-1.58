"""Low-Rank Adaptation (LoRA) for efficient fine-tuning of BitVideo models."""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-rank adaptation layer that wraps a linear projection.

    Adds a low-rank decomposition ``BA`` to the original weight where
    ``A`` has shape ``[rank, in_features]`` and ``B`` has shape
    ``[out_features, rank]``. During training only A and B are updated;
    the original weight is frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        # A: [rank, in_features]; B: [out_features, rank].
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * (x @ A^T @ B^T).

        Args:
            x: Input ``[..., in_features]``.

        Returns:
            LoRA contribution ``[..., out_features]``.
        """
        # dropped: [..., in_features].
        dropped = self.dropout(x)
        # delta: [..., out_features] = x @ A^T @ B^T * scaling.
        return F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}"
        )


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: tuple[str, ...] = ("query_projection", "value_projection"),
    dropout: float = 0.0,
) -> dict[str, LoRALayer]:
    """Apply LoRA layers to target linear modules in a model.

    Freezes the base model parameters and attaches trainable LoRA layers
    as attributes. Returns a dict of the created LoRA layers.

    Args:
        model: The model to adapt.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        target_modules: Names of submodules to wrap.
        dropout: Dropout rate for LoRA layers.

    Returns:
        Dict mapping qualified module names to LoRA layers.
    """
    # Freeze all existing parameters.
    for param in model.parameters():
        param.requires_grad = False

    lora_layers: dict[str, LoRALayer] = {}
    # Collect targets first to avoid modifying dict during iteration.
    targets: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        module_name = name.split(".")[-1] if "." in name else name
        if module_name in target_modules:
            if hasattr(module, "weight") and module.weight is not None:
                targets.append((name, module))

    for name, module in targets:
        in_f = module.weight.shape[1]
        out_f = module.weight.shape[0]
        lora = LoRALayer(
            in_f, out_f, rank=rank, alpha=alpha, dropout=dropout,
            device=module.weight.device, dtype=module.weight.dtype,
        )
        # Register as a submodule of the parent.
        parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
        parent = model.get_submodule(parent_name) if parent_name else model
        module_name = name.split(".")[-1]
        attr_name = f"lora_{module_name}"
        parent.register_module(attr_name, lora)
        lora_layers[name] = lora

    return lora_layers


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRA weights into the base model weights permanently.

    After merging, LoRA layers can be removed and the model runs without
    any inference overhead.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALayer):
            # Find the corresponding base linear layer.
            parts = name.split(".")
            # The LoRA module name is "lora_<target>", so the target is nearby.
            parent_name = ".".join(parts[:-1])
            parent = model.get_submodule(parent_name) if parent_name else model
            target_name = parts[-1].replace("lora_", "")
            if hasattr(parent, target_name):
                target_module = getattr(parent, target_name)
                if hasattr(target_module, "weight"):
                    # delta_weight: [out, in] = B @ A * scaling.
                    delta = (module.lora_B @ module.lora_A) * module.scaling
                    with torch.no_grad():
                        target_module.weight.add_(delta.to(target_module.weight.dtype))
                    # Remove the LoRA module.
                    delattr(parent, parts[-1])
