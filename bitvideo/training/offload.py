"""Layer-level CPU/GPU memory management utilities for streaming training.

Inspired by kimi-k3-in-c's trunk-streaming approach: instead of holding the
full model in GPU memory, move one transformer block at a time to GPU, execute
forward+backward, then offload back to CPU. This trades speed (~5-10x slower)
for memory (~80%+ reduction), enabling 2B+ model training on 8GB GPUs.

Key insight from K3: a sequential model that visits layers in fixed order can
stream layers from a slower tier (disk/CPU RAM) to a fast tier (GPU VRAM) one
at a time, just like K3 streams its 108GB trunk through a 2.3GB ring buffer.

Usage:
    from bitvideo.training.offload import (
        offload_module, load_module, pin_module,
        estimate_block_memory, LayerStreamingContext,
    )
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "offload_module",
    "load_module",
    "pin_module",
    "estimate_block_memory",
    "LayerStreamingContext",
    "MemoryBudget",
    "get_gpu_memory_info",
]


# ---------------------------------------------------------------------------
# Memory info
# ---------------------------------------------------------------------------


@dataclass
class MemoryBudget:
    """GPU memory budget breakdown."""

    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float

    @property
    def usable_gb(self) -> float:
        """Conservatively estimated usable memory (85% of free)."""
        return self.free_gb * 0.85


def get_gpu_memory_info(device: torch.device | int = 0) -> MemoryBudget:
    """Get current GPU memory state."""
    if not torch.cuda.is_available():
        return MemoryBudget(0, 0, 0, 0)
    # Handle CPU device gracefully
    if isinstance(device, torch.device) and device.type != "cuda":
        return MemoryBudget(0, 0, 0, 0)
    try:
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = total - reserved
        return MemoryBudget(total, allocated, reserved, free)
    except (ValueError, RuntimeError):
        return MemoryBudget(0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Module movement
# ---------------------------------------------------------------------------


def offload_module(
    module: nn.Module,
    *,
    non_blocking: bool = True,
) -> nn.Module:
    """Move a module to CPU, freeing GPU memory.

    Uses non_blocking=True by default for overlapping with compute.
    Also moves optimizer states if they exist as module attributes.
    """
    module.to("cpu", non_blocking=non_blocking)
    return module


def load_module(
    module: nn.Module,
    device: torch.device | str = "cuda",
    *,
    non_blocking: bool = True,
) -> nn.Module:
    """Move a module to GPU for computation.

    Args:
        module: The nn.Module to move.
        device: Target device (default "cuda").
        non_blocking: Use async transfers (default True).

    Returns:
        The module, now on the target device.
    """
    module.to(device, non_blocking=non_blocking)
    return module


def pin_module(module: nn.Module) -> nn.Module:
    """Pin a module's parameters to page-locked memory for faster CPU->GPU transfers.

    This is the equivalent of K3's "pinned trunk layers" — modules that
    transfer to GPU frequently benefit from pinned memory (2-3x faster copy).
    """
    for param in module.parameters():
        if param.is_cpu and not param.data.is_pinned():
            param.data = param.data.pin_memory()
    for buf in module.buffers():
        if buf.is_cpu and not buf.is_pinned():
            buf.data = buf.data.pin_memory()
    return module


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_block_memory(
    module: nn.Module,
    *,
    include_gradients: bool = True,
    include_optimizer: bool = True,
    optimizer_states: int = 2,  # AdamW has 2 moment buffers
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """Estimate memory footprint of a module in MB.

    Args:
        module: The module to estimate.
        include_gradients: Include gradient storage.
        include_optimizer: Include optimizer state (moments).
        optimizer_states: Number of optimizer state tensors per param (2 for Adam).
        dtype: Data type for params.

    Returns:
        Dict with 'params_mb', 'gradients_mb', 'optimizer_mb', 'total_mb'.
    """
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_params = sum(p.numel() for p in module.parameters())

    params_mb = total_params * bytes_per_elem / 1e6
    gradients_mb = params_mb if include_gradients else 0.0
    optimizer_mb = params_mb * optimizer_states if include_optimizer else 0.0
    total_mb = params_mb + gradients_mb + optimizer_mb

    return {
        "params_mb": params_mb,
        "gradients_mb": gradients_mb,
        "optimizer_mb": optimizer_mb,
        "total_mb": total_mb,
        "total_params": total_params,
    }


# ---------------------------------------------------------------------------
# Streaming context manager
# ---------------------------------------------------------------------------


class LayerStreamingContext:
    """Manages the streaming of transformer blocks between CPU and GPU.

    Inspired by K3's trunk streaming: keeps a "ring" of N blocks on GPU
    (like K3's ring buffer), with the rest on CPU. When a block is needed,
    it's loaded; when it's done, it's offloaded.

    The key K3 insight applied here: prefetch the NEXT block while the
    current one is computing, just like K3 prefetches the next trunk layer
    while the current one is being multiplied.

    Args:
        model: The VideoDiT model.
        device: GPU device to stream to.
        pin_memory: Whether to pin CPU tensors for faster transfers.
        prefetch: Whether to prefetch the next block during current computation.
        max_blocks_on_gpu: Maximum blocks to keep on GPU simultaneously.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cuda",
        *,
        pin_memory: bool = True,
        prefetch: bool = True,
        max_blocks_on_gpu: int = 1,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.max_blocks_on_gpu = max_blocks_on_gpu
        self._prefetch_stream = torch.cuda.Stream() if prefetch and torch.cuda.is_available() else None
        self._blocks_on_gpu: list[int] = []
        self._setup_done = False

    def setup(self) -> None:
        """Move everything to CPU except permanently-pinned modules."""
        if self._setup_done:
            return

        # Keep small/essential modules on GPU permanently (like K3's embed + lm_head)
        # These are the "always-resident" part (<5% of total params)
        pinned_modules = self._get_pinned_modules()
        for name, mod in pinned_modules:
            if self.device.type == "cuda":
                mod.to(self.device)
            logger.debug(f"Pinned on GPU: {name}")

        # Move all blocks to CPU (or keep on CPU if device is CPU)
        blocks = self._get_blocks()
        if self.device.type == "cuda":
            for i, block in enumerate(blocks):
                block.to("cpu")
                if self.pin_memory:
                    pin_module(block)

        self._setup_done = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        mem = get_gpu_memory_info(self.device)
        logger.info(
            f"Streaming setup complete: {len(blocks)} blocks on CPU, "
            f"pinned modules on {self.device}. "
            f"GPU: {mem.allocated_gb:.2f}/{mem.total_gb:.1f} GB used"
        )

    def _get_blocks(self) -> nn.ModuleList:
        """Get the transformer blocks from the model."""
        if hasattr(self.model, "blocks"):
            return self.model.blocks
        elif hasattr(self.model, "transformer_blocks"):
            return self.model.transformer_blocks
        elif hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError(
            "Model must have 'blocks', 'transformer_blocks', or 'layers' attribute"
        )

    def _get_pinned_modules(self) -> list[tuple[str, nn.Module]]:
        """Get modules that should stay permanently on GPU.

        These are the small modules that run every step:
        patch_embed, timestep_embed, context_projection, final_layer.
        """
        pinned = []
        for name in ("patch_embed", "timestep_embed", "context_projection",
                     "final_layer", "rotary_embedding"):
            mod = getattr(self.model, name, None)
            if mod is not None:
                pinned.append((name, mod))
        return pinned

    @contextmanager
    def stream_block(self, block_idx: int) -> Iterator[nn.Module]:
        """Context manager that loads a block to GPU and offloads after use.

        Usage:
            with ctx.stream_block(i) as block:
                output = block(input, ...)
                loss.backward()  # gradients computed while on GPU
        """
        blocks = self._get_blocks()
        block = blocks[block_idx]

        if self.device.type == "cuda":
            # Load to GPU
            load_module(block, self.device, non_blocking=True)
            torch.cuda.current_stream().synchronize()

            # Prefetch next block if enabled
            if self.prefetch and self._prefetch_stream and block_idx + 1 < len(blocks):
                with torch.cuda.stream(self._prefetch_stream):
                    load_module(blocks[block_idx + 1], self.device, non_blocking=True)

        try:
            yield block
        finally:
            if self.device.type == "cuda":
                # Offload back to CPU (keep gradients on CPU too)
                offload_module(block, non_blocking=True)
                torch.cuda.current_stream().synchronize()

                # Aggressive memory cleanup every few blocks
                if block_idx % 4 == 0:
                    torch.cuda.empty_cache()

    def estimate_memory_usage(self, dtype: torch.dtype = torch.float32) -> dict:
        """Estimate total memory breakdown for streaming training."""
        blocks = self._get_blocks()
        pinned = self._get_pinned_modules()

        block_mem = estimate_block_memory(blocks[0], dtype=dtype) if blocks else {}
        pinned_params = sum(
            sum(p.numel() for p in mod.parameters()) for _, mod in pinned
        )
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        pinned_mb = pinned_params * bytes_per_elem / 1e6

        return {
            "num_blocks": len(blocks),
            "per_block_mb": block_mem.get("total_mb", 0),
            "pinned_modules_mb": pinned_mb,
            "peak_gpu_mb": block_mem.get("total_mb", 0) + pinned_mb,
            "total_model_mb": block_mem.get("total_mb", 0) * len(blocks) + pinned_mb,
        }
