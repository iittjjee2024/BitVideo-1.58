"""BitVideo Distillation Training CLI.

Train BitVideo student model by distilling LTX-2.3 teacher knowledge.

Usage:
    # Standard distillation (A100 80GB)
    python scripts/train_distill.py --student-dim 3072 --student-depth 32 \\
        --teacher-data data/teacher_latents --max-steps 100000

    # Smaller student (A100 40GB or H100)
    python scripts/train_distill.py --student-dim 2048 --student-depth 24 \\
        --teacher-data data/teacher_latents --max-steps 50000

    # Budget distillation (T4 16GB with streaming)
    python scripts/train_distill.py --student-dim 1536 --student-depth 20 \\
        --teacher-data data/teacher_latents --streaming --max-steps 50000

    # Resume
    python scripts/train_distill.py --resume checkpoints/distilled/distill_step_50000.pt
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitvideo.training.distill_ltx import (
    DistillConfig,
    DistillationTrainer,
    TeacherStudentDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="BitVideo distillation training")

    # Student architecture
    parser.add_argument("--student-dim", type=int, default=3072)
    parser.add_argument("--student-depth", type=int, default=32)
    parser.add_argument("--student-heads", type=int, default=24)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--in-channels", type=int, default=128)

    # Data
    parser.add_argument("--teacher-data", type=str, required=True,
                        help="Path to teacher latents directory")

    # Training
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2000)

    # Losses
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.05)
    parser.add_argument("--feature-weight", type=float, default=0.1)

    # Memory
    parser.add_argument("--streaming", action="store_true",
                        help="Use layer streaming (for T4/low-memory GPUs)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16")

    # Checkpoint
    parser.add_argument("--output-dir", type=str, default="checkpoints/distilled")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Print info
    print("=" * 60)
    print("BitVideo DISTILLATION — LTX-2.3 → BitVideo")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    approx_params = args.student_dim * args.student_dim * 12 * args.student_depth
    print(f"Student: {args.student_dim}D x {args.student_depth} layers = ~{approx_params/1e9:.1f}B params")
    print(f"Teacher data: {args.teacher_data}")
    print(f"Streaming: {'ON' if args.streaming else 'OFF'}")
    print("=" * 60)

    # Config
    config = DistillConfig(
        student_dim=args.student_dim,
        student_depth=args.student_depth,
        student_heads=args.student_heads,
        student_context_dim=args.context_dim,
        student_in_channels=args.in_channels,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup,
        mse_weight=args.mse_weight,
        temporal_weight=args.temporal_weight,
        feature_weight=args.feature_weight,
        teacher_data_dir=args.teacher_data,
        use_streaming=args.streaming,
        dtype=args.dtype,
        output_dir=args.output_dir,
        save_every_steps=args.save_every,
        log_every_steps=args.log_every,
        resume_from=args.resume,
    )

    # Dataset
    dataset = TeacherStudentDataset(args.teacher_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # Train
    trainer = DistillationTrainer(config)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
