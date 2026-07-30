"""Configuration management for BitVideo-1.58 models and pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class BitVideoConfig:
    """Complete configuration for a BitVideo-1.58 model instance.

    Stores all hyperparameters needed to instantiate a VideoDiT model,
    its scheduler, and the training setup. Can be serialized to/from JSON.
    """

    # Model architecture.
    in_channels: int = 4
    out_channels: int = 4
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    head_dim: int | None = None
    num_kv_heads: int | None = None
    context_dim: int = 768
    patch_size: tuple[int, ...] = (1, 2, 2)
    ffn_expansion_ratio: float = 4.0
    ffn_activation: str = "swiglu"

    # Attention.
    qkv_bias: bool = True
    qk_norm: bool = True
    attention_dropout: float = 0.0

    # Diffusion.
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"

    # Quantization.
    weight_bits: float = 1.58
    activation_bits: int = 8
    activation_granularity: str = "per_token"

    # RoPE.
    rope_base: float = 10_000.0
    rope_axis_dims: tuple[int, ...] | None = None

    # Training.
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # Inference.
    default_num_inference_steps: int = 50
    default_guidance_scale: float = 7.5

    # Metadata.
    model_name: str = "bitvideo-1.58"
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BitVideoConfig":
        """Deserialize from a dict, ignoring unknown keys."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        # Convert lists back to tuples for tuple fields.
        for key in ("patch_size", "rope_axis_dims"):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        return cls(**filtered)


def save_config(config: BitVideoConfig, path: str | Path) -> Path:
    """Save configuration to a JSON file.

    Args:
        config: Configuration to save.
        path: Output file path.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    return path


def load_config(path: str | Path) -> BitVideoConfig:
    """Load configuration from a JSON file.

    Args:
        path: Path to the config JSON.

    Returns:
        Loaded BitVideoConfig.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return BitVideoConfig.from_dict(data)
