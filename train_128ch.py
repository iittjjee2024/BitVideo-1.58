"""BitVideo-1.58 Training with 128-channel LTX VAE latents.

Run after encode_with_ltx_vae.py completes.
Output is decodable by the LTX Video VAE into real video frames.

Usage: python train_128ch.py
"""

import os
import sys
import json
import time
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Paths
DATA_DIR = r"C:\bitvideo_data_128ch"
OUTPUT_DIR = r"G:\My Drive\Production-Grade-Projects\1-bit\outputs_128ch"

# Model config — 128 latent channels to match LTX VAE
DIM = 128
DEPTH = 4
NUM_HEADS = 4
IN_CHANNELS = 128       # LTX VAE latent channels
CONTEXT_DIM = 768
PATCH_SIZE = (1, 2, 2)  # Temporal, Height, Width patches

# Training
BATCH_SIZE = 1
MAX_STEPS = 3000
LEARNING_RATE = 1e-4
MIXED_PRECISION = "bf16"
SAVE_EVERY = 1000
LOG_EVERY = 50

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitvideo.models import VideoDiT
from bitvideo.training.train_qat_video import TrainingConfig, QATTrainer
from bitvideo.training.datasets import create_dataloader


class UTF8Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / "metadata.json", "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        logging.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        video = torch.load(self.root / s["video"], map_location="cpu", weights_only=True)
        text = torch.load(self.root / s["text"], map_location="cpu", weights_only=True)
        return {"video_latent": video, "text_embedding": text}


def main():
    print("=" * 60)
    print("  BitVideo-1.58 Training (128ch LTX VAE Latents)")
    print("=" * 60)
    print(f"  Data: {DATA_DIR}")
    print(f"  Model: dim={DIM}, depth={DEPTH}, in_channels={IN_CHANNELS}")
    print(f"  Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}")
    print("=" * 60)

    # Verify data exists
    meta_path = Path(DATA_DIR) / "metadata.json"
    if not meta_path.exists():
        print("ERROR: No encoded data found. Run encode_with_ltx_vae.py first.")
        return

    # Load dataset
    dataset = UTF8Dataset(DATA_DIR)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Check latent shape from first sample
    sample = dataset[0]
    latent_shape = sample["video_latent"].shape
    print(f"  Latent shape: {tuple(latent_shape)}")
    num_frames = latent_shape[1]  # Temporal dimension
    height = latent_shape[2]
    width = latent_shape[3]

    # Configure training
    config = TrainingConfig(
        dim=DIM, depth=DEPTH, num_heads=NUM_HEADS,
        context_dim=CONTEXT_DIM, in_channels=IN_CHANNELS,
        patch_size=PATCH_SIZE,
        num_frames=num_frames, height=height, width=width,
        text_length=77, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, max_steps=MAX_STEPS,
        warmup_steps=min(200, MAX_STEPS // 10),
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        prediction_type="epsilon", loss_type="mse", snr_gamma=5.0,
        num_train_timesteps=1000, beta_schedule="linear",
        mixed_precision=MIXED_PRECISION,
        output_dir=OUTPUT_DIR,
        save_every_steps=SAVE_EVERY, log_every_steps=LOG_EVERY,
        seed=42, num_workers=0,
    )

    # Train
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer = QATTrainer(config)
    print(f"  Parameters: {trainer.model.parameter_count():,}")
    print(f"  Effective batch: {BATCH_SIZE * config.gradient_accumulation_steps}")
    print()

    trainer.train(dataloader)

    print(f"\nTraining complete! Checkpoint: {OUTPUT_DIR}")
    print("Next: run app_real_video.py to generate with VAE decoder")


if __name__ == "__main__":
    main()
