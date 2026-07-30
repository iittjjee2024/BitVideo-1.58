"""Dataset utilities for video-text paired training data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class VideoTextDataset(Dataset):
    """Dataset for video-text pairs stored as pre-extracted latents.

    Expects a directory structure:
        root/
            metadata.json   (list of {"video": "path.pt", "text": "path.pt"})
            latents/
                video_0000.pt  (tensor [C, T, H, W])
                text_0000.pt   (tensor [L, D])

    This avoids video decoding during training by operating entirely on
    pre-extracted VAE latents and text encoder embeddings.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        metadata_file: str = "metadata.json",
        max_samples: int | None = None,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        self.transform = transform

        metadata_path = self.root / metadata_file
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            # Auto-discover paired files: video_XXXX.pt + text_XXXX.pt
            latent_dir = self.root / "latents"
            if not latent_dir.exists():
                latent_dir = self.root
            video_files = sorted(latent_dir.glob("video_*.pt"))
            self.samples = []
            for vf in video_files:
                idx = vf.stem.replace("video_", "")
                tf = vf.parent / f"text_{idx}.pt"
                if tf.exists():
                    self.samples.append({
                        "video": str(vf.relative_to(self.root)),
                        "text": str(tf.relative_to(self.root)),
                    })

        if max_samples is not None:
            max_samples = _positive_int(max_samples, "max_samples")
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load a video-text pair.

        Returns:
            Dict with 'video_latent' [C, T, H, W] and 'text_embedding' [L, D].
        """

        if index < 0 or index >= len(self.samples):
            raise IndexError(f"index {index} out of range [0, {len(self.samples)})")
        sample = self.samples[index]
        video_path = self.root / sample["video"]
        text_path = self.root / sample["text"]

        # video_latent: [C, T, H, W]; text_embedding: [L, D].
        video_latent = torch.load(video_path, map_location="cpu", weights_only=True)
        text_embedding = torch.load(text_path, map_location="cpu", weights_only=True)

        result = {"video_latent": video_latent, "text_embedding": text_embedding}
        if self.transform is not None:
            result = self.transform(result)
        return result


class SyntheticVideoDataset(Dataset):
    """Synthetic dataset for testing and debugging training loops.

    Generates random latents and text embeddings on-the-fly without any
    disk I/O. Useful for validating the training pipeline.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        *,
        latent_channels: int = 4,
        num_frames: int = 16,
        height: int = 32,
        width: int = 32,
        text_length: int = 77,
        text_dim: int = 768,
        seed: int = 42,
    ) -> None:
        self.num_samples = _positive_int(num_samples, "num_samples")
        self.latent_channels = _positive_int(latent_channels, "latent_channels")
        self.num_frames = _positive_int(num_frames, "num_frames")
        self.height = _positive_int(height, "height")
        self.width = _positive_int(width, "width")
        self.text_length = _positive_int(text_length, "text_length")
        self.text_dim = _positive_int(text_dim, "text_dim")
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Generate a synthetic video-text pair.

        Returns:
            Dict with 'video_latent' [C, T, H, W] and 'text_embedding' [L, D].
        """

        if index < 0 or index >= self.num_samples:
            raise IndexError(f"index {index} out of range [0, {self.num_samples})")
        # Deterministic generation per index.
        gen = torch.Generator().manual_seed(self.seed + index)
        video_latent = torch.randn(
            self.latent_channels, self.num_frames, self.height, self.width, generator=gen
        )
        text_embedding = torch.randn(self.text_length, self.text_dim, generator=gen)
        return {"video_latent": video_latent, "text_embedding": text_embedding}


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for video training.

    Args:
        dataset: The dataset to load from.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle each epoch.
        num_workers: Number of data loading workers.
        pin_memory: Pin host memory for faster GPU transfer.
        drop_last: Drop the last incomplete batch.
        prefetch_factor: Batches to prefetch per worker.

    Returns:
        Configured DataLoader.
    """

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
