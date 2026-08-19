"""BitVideo-1.58 Streaming Training Script — Train 2B models on 8GB+ GPUs.

This script uses the layer-streaming approach inspired by kimi-k3-in-c:
instead of holding the full model in GPU memory, it streams one transformer
block at a time between CPU RAM and GPU VRAM.

Requirements:
    - GPU: 8GB+ VRAM (T4, RTX 3060, RTX 4060, etc.)
    - CPU RAM: 32GB+ (holds model + optimizer states)
    - Fast SSD recommended (NVMe helps with checkpointing)

Memory comparison (2B model, dim=2048, depth=24):
    Standard training:  ~16 GB GPU
    Streaming training: ~2-3 GB GPU + 32 GB CPU RAM

Speed comparison:
    Standard: ~5-10 steps/sec
    Streaming: ~0.1-0.5 steps/sec (5-50x slower)

Usage:
    # 2B model on T4 16GB (streaming)
    python scripts/train_streaming.py --dim 2048 --depth 24

    # 1B model on RTX 3060 12GB
    python scripts/train_streaming.py --dim 1536 --depth 20

    # 500M model on 8GB GPU (minimum)
    python scripts/train_streaming.py --dim 1024 --depth 16

    # Resume from checkpoint
    python scripts/train_streaming.py --dim 2048 --depth 24 --resume checkpoints/streaming_ckpt_5000.pt

    # With pre-encoded latents
    python scripts/train_streaming.py --dim 2048 --depth 24 --data-dir /path/to/encoded
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitvideo.training.streaming import StreamingTrainer, StreamingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EncodedLatentDataset(Dataset):
    """Load pre-encoded video latents + text embeddings from disk.

    Expected structure:
        data_dir/
            metadata.json     <- list of {latent: path, condition: path}
            latents/          <- video latent .pt files
            conditions/       <- text embedding .pt files
    """

    def __init__(self, data_dir: str):
        self.root = Path(data_dir)
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            # Auto-build metadata from latent/condition file pairs
            latents = sorted((self.root / "latents").glob("*.pt"))
            conditions = sorted((self.root / "conditions").glob("*.pt"))
            self.samples = [
                {"latent": f"latents/{l.name}", "condition": f"conditions/{c.name}"}
                for l, c in zip(latents, conditions)
            ]
            logger.info(f"Auto-built metadata: {len(self.samples)} samples")
        else:
            with open(meta_path, "r") as f:
                self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        lat_data = torch.load(self.root / s["latent"], weights_only=True)
        cond_data = torch.load(self.root / s["condition"], weights_only=True)

        # Handle both dict and raw tensor formats
        video = lat_data["latent"] if isinstance(lat_data, dict) else lat_data
        text = cond_data["embedding"] if isinstance(cond_data, dict) else cond_data

        return {"video_latent": video, "text_embedding": text}


class SyntheticDataset(Dataset):
    """Synthetic random data for testing streaming training without real data."""

    def __init__(
        self,
        num_samples: int = 1000,
        in_channels: int = 128,
        temporal: int = 3,
        height: int = 4,
        width: int = 4,
        context_dim: int = 1024,
        context_len: int = 77,
    ):
        self.num_samples = num_samples
        self.video_shape = (in_channels, temporal, height, width)
        self.text_shape = (context_len, context_dim)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "video_latent": torch.randn(*self.video_shape),
            "text_embedding": torch.randn(*self.text_shape),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BitVideo-1.58 Streaming Training (low-memory GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2B model on T4 (16GB GPU, 32GB+ CPU RAM)
  python scripts/train_streaming.py --dim 2048 --depth 24 --data-dir /path/to/encoded

  # 1B model on RTX 3060 (12GB)
  python scripts/train_streaming.py --dim 1536 --depth 20 --data-dir /path/to/encoded

  # Quick test with synthetic data
  python scripts/train_streaming.py --dim 512 --depth 4 --synthetic --max-steps 10

  # Resume training
  python scripts/train_streaming.py --dim 2048 --depth 24 --resume checkpoints/streaming_ckpt_5000.pt
""",
    )

    # Model
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension (default: 2048 = 2B)")
    parser.add_argument("--depth", type=int, default=24, help="Number of transformer blocks (default: 24)")
    parser.add_argument("--num-heads", type=int, default=16, help="Attention heads (default: 16)")
    parser.add_argument("--in-channels", type=int, default=128, help="Input channels (default: 128 for LTX VAE)")
    parser.add_argument("--context-dim", type=int, default=1024, help="Text encoder dim (default: 1024 for T5-Large)")

    # Data
    parser.add_argument("--data-dir", type=str, default=None, help="Path to encoded latents directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data (for testing)")

    # Training
    parser.add_argument("--max-steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=5000, help="LR warmup steps")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep 1 for streaming)")

    # Streaming
    parser.add_argument("--max-gpu-blocks", type=int, default=1,
                        help="Max blocks on GPU at once (1=min memory, more=faster)")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable CPU memory pinning")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable next-block prefetching")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing (uses more GPU memory)")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Master weight dtype")

    return parser.parse_args()


def main():
    args = parse_args()

    # Print banner
    print("=" * 60)
    print("BitVideo-1.58 — STREAMING TRAINER")
    print("Train 2B+ models on 8GB+ GPUs via layer streaming")
    print("=" * 60)

    # Check GPU
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif args.device == "cuda":
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    else:
        print(f"Device: {args.device}")

    # Estimate model size
    approx_params = args.dim * args.dim * 12 * args.depth  # rough estimate
    print(f"Model: dim={args.dim}, depth={args.depth}, heads={args.num_heads}")
    print(f"Estimated: ~{approx_params/1e9:.1f}B params")
    print(f"Streaming: {args.max_gpu_blocks} block(s) on GPU at a time")
    print(f"Gradient checkpointing: {'OFF' if args.no_grad_checkpoint else 'ON'}")
    print(f"Memory pinning: {'OFF' if args.no_pin_memory else 'ON'}")
    print(f"Prefetch: {'OFF' if args.no_prefetch else 'ON'}")
    print("=" * 60)

    # Create config
    config = StreamingConfig(
        # Model
        in_channels=args.in_channels,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        context_dim=args.context_dim,
        # Training
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        gradient_accumulation_steps=args.grad_accum,
        batch_size=args.batch_size,
        # Streaming
        max_gpu_blocks=args.max_gpu_blocks,
        pin_cpu_memory=not args.no_pin_memory,
        gradient_checkpointing=not args.no_grad_checkpoint,
        prefetch_next_block=not args.no_prefetch,
        # Checkpoint
        output_dir=args.output_dir,
        save_every_steps=args.save_every,
        log_every_steps=args.log_every,
        resume_from=args.resume,
        # Device
        device=args.device,
        dtype=args.dtype,
    )

    # Create trainer
    print("\nInitializing streaming trainer...")
    t0 = time.time()
    trainer = StreamingTrainer(config)
    print(f"  Init time: {time.time() - t0:.1f}s")

    # Create dataset
    if args.synthetic:
        print("\nUsing SYNTHETIC data (for testing)")
        dataset = SyntheticDataset(
            num_samples=1000,
            in_channels=args.in_channels,
            context_dim=args.context_dim,
        )
    elif args.data_dir:
        print(f"\nLoading data from: {args.data_dir}")
        dataset = EncodedLatentDataset(args.data_dir)
        print(f"  Samples: {len(dataset)}")
    else:
        print("\nNo --data-dir specified. Using synthetic data for demo.")
        print("  For real training, provide --data-dir /path/to/encoded")
        dataset = SyntheticDataset(
            num_samples=1000,
            in_channels=args.in_channels,
            context_dim=args.context_dim,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if not args.synthetic else 0,
        drop_last=True,
        pin_memory=True,
    )

    # Train
    print(f"\nStarting training: step {trainer.global_step} -> {args.max_steps}")
    print(f"  Speed estimate: ~0.1-0.5 steps/sec (streaming is slow but fits!)")
    print(f"  ETA for 50K steps: ~28-140 hours")
    print("")

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
