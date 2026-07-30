"""BitVideo-1.58 Training Script — Run directly with: python train.py"""

import sys
import os
import json
import time
import logging
import torch
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = r"C:\bitvideo_data"
OUTPUT_DIR = r"G:\My Drive\Production-Grade-Projects\1-bit\outputs"

# ============================================================
# CONFIGURATION — EDIT THESE
# ============================================================
DIM = 128              # Model hidden dimension (256 for medium, 768 for full)
DEPTH = 4             # Transformer blocks (6 for medium, 12 for full)
NUM_HEADS = 4         # Attention heads
BATCH_SIZE = 1        # Reduce to 1 if OOM
MAX_STEPS = 2000      # Total training steps
LEARNING_RATE = 1e-4
MIXED_PRECISION = "bf16"  # "bf16", "fp16", or "none"
SAVE_EVERY = 1000
LOG_EVERY = 50
# ============================================================

from bitvideo.training.datasets import create_dataloader
from bitvideo.training.train_qat_video import TrainingConfig, QATTrainer


class UTF8VideoDataset(torch.utils.data.Dataset):
    """Dataset that handles UTF-8 metadata correctly on Windows."""

    def __init__(self, root: str):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        logger.info(f"Loaded {len(self.samples)} samples from {metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        video_latent = torch.load(
            self.root / sample["video"], map_location="cpu", weights_only=True
        )
        text_embedding = torch.load(
            self.root / sample["text"], map_location="cpu", weights_only=True
        )
        return {"video_latent": video_latent, "text_embedding": text_embedding}


def main():
    print("=" * 60)
    print("  BitVideo-1.58 Training")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Data: {DATA_DIR}")
    print(f"  Model: dim={DIM}, depth={DEPTH}, heads={NUM_HEADS}")
    print(f"  Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Mixed precision: {MIXED_PRECISION}")
    print("=" * 60)

    # Load dataset
    dataset = UTF8VideoDataset(DATA_DIR)
    dataloader = create_dataloader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    # Configure training
    config = TrainingConfig(
        dim=DIM, depth=DEPTH, num_heads=NUM_HEADS, context_dim=768,
        in_channels=4, patch_size=(1, 2, 2),
        num_frames=16, height=32, width=32, text_length=77,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS, warmup_steps=min(500, MAX_STEPS // 10),
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        prediction_type="epsilon", loss_type="mse", snr_gamma=5.0,
        num_train_timesteps=1000, beta_schedule="linear",
        mixed_precision=MIXED_PRECISION,
        output_dir=OUTPUT_DIR,
        save_every_steps=SAVE_EVERY, log_every_steps=LOG_EVERY,
        seed=42, num_workers=0,
    )

    # Initialize and train
    trainer = QATTrainer(config)
    print(f"\n  Parameters: {trainer.model.parameter_count():,}")
    print(f"  Effective batch: {BATCH_SIZE * config.gradient_accumulation_steps}")
    print()

    trainer.train(dataloader)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Checkpoint: {OUTPUT_DIR}/checkpoint-{trainer.global_step}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
