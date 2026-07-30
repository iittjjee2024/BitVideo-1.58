"""Encode video clips into LTX VAE latents (128 channels) for BitVideo training.

This script:
1. Loads the LTX Video VAE encoder
2. Reads your video clips
3. Encodes them into proper 128-channel latents
4. Saves as .pt files ready for BitVideo training

The resulting latents can be decoded by the same VAE to produce real video.

Usage: python scripts/encode_with_ltx_vae.py
"""

import os
import sys
import json
import re
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Config
CLIPS_DIR = r"G:\My Drive\Antigravity_Production\Training_Output\clips"
OUTPUT_DIR = r"C:\bitvideo_data_128ch"
VAE_CACHE = r"C:\bitvideo_vae"

# VAE output format: [128, T_latent, H_latent, W_latent]
# With input [3, T*8+1, H*32, W*32] -> latent [128, T, H, W]
# So for 17 frames at 128x128 pixels: latent = [128, 2, 4, 4]
# For 9 frames at 64x64: latent = [128, 1, 2, 2] (too small)
# Let's target: 17 frames at 256x256 -> needs [3, 17, 256, 256] input -> [128, 2, 8, 8] latent

TARGET_FRAMES = 17      # Must be 8k+1 (9, 17, 25, 33...)
TARGET_HEIGHT = 128     # Pixel height for VAE input
TARGET_WIDTH = 128      # Pixel width for VAE input
TEXT_DIM = 768
TEXT_LENGTH = 77
MAX_CLIPS = None        # None = all, or set a number for testing
BATCH_SIZE = 1          # VAE encode batch size (1 to save VRAM)

print("=" * 60)
print("  LTX VAE Encoding Pipeline")
print("=" * 60)
print(f"  Clips: {CLIPS_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Target: {TARGET_FRAMES} frames @ {TARGET_HEIGHT}x{TARGET_WIDTH}")
print("=" * 60)

# Load VAE
print("\nLoading LTX Video VAE...")
from diffusers.models import AutoencoderKLLTXVideo

vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video", subfolder="vae",
    torch_dtype=torch.float16, cache_dir=VAE_CACHE,
)
vae = vae.to("cuda").eval()
print(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()):,} params")

# Test encode to determine output shape
with torch.no_grad():
    test_input = torch.randn(1, 3, TARGET_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, device="cuda", dtype=torch.float16)
    test_latent = vae.encode(test_input).latent_dist.sample()
    LATENT_C, LATENT_T, LATENT_H, LATENT_W = test_latent.shape[1:]
    del test_input, test_latent
    torch.cuda.empty_cache()

print(f"  Latent shape: [{LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W}]")
print(f"  (128 channels, {LATENT_T} temporal, {LATENT_H}x{LATENT_W} spatial)")

# Find clips
print("\nScanning clips...")
import cv2

all_clips = sorted(Path(CLIPS_DIR).rglob("*.mp4"))
if MAX_CLIPS:
    all_clips = all_clips[:MAX_CLIPS]
print(f"  Found {len(all_clips)} clips")

# Create output dir
os.makedirs(os.path.join(OUTPUT_DIR, "latents"), exist_ok=True)


def load_video_frames(clip_path, num_frames, height, width):
    """Load and preprocess video frames for VAE encoding."""
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Sample frames uniformly
    indices = torch.linspace(0, total - 1, num_frames).long().tolist()
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            # Normalize to [-1, 1] (VAE expects this)
            frame_tensor = torch.from_numpy(frame).float() / 127.5 - 1.0
            frames.append(frame_tensor)
        elif frames:
            frames.append(frames[-1].clone())
        else:
            frames.append(torch.zeros(height, width, 3))

    cap.release()

    if len(frames) != num_frames:
        return None

    # Stack: [T, H, W, 3] -> [3, T, H, W]
    video = torch.stack(frames).permute(3, 0, 1, 2)
    return video


def caption_from_path(clip_path):
    """Extract caption from folder name."""
    folder = clip_path.parent.name
    caption = re.sub(r"\(?\d+[PKpk]_?(?:HD|60FPS)?\)?", "", folder)
    caption = re.sub(r"[_\-]+", " ", caption).strip()
    return caption if caption else "a video clip"


def text_embedding_from_caption(caption):
    """Create pseudo text embedding (seeded from caption)."""
    seed = hash(caption) % (2**31)
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(TEXT_LENGTH, TEXT_DIM, generator=gen) * 0.1
    for i, ch in enumerate(caption[:TEXT_LENGTH]):
        emb[i, 0] = (ord(ch) / 128.0 - 1.0) * 0.5
    return emb


# Encode all clips
print("\nEncoding clips with LTX VAE...")
metadata = []
errors = []
t_start = time.time()

for idx, clip_path in enumerate(all_clips):
    try:
        # Load video
        video = load_video_frames(clip_path, TARGET_FRAMES, TARGET_HEIGHT, TARGET_WIDTH)
        if video is None:
            errors.append(f"Could not read: {clip_path.name}")
            continue

        # Encode with VAE: [1, 3, T, H, W] -> [1, 128, T_lat, H_lat, W_lat]
        video_input = video.unsqueeze(0).to("cuda", dtype=torch.float16)

        with torch.no_grad():
            latent_dist = vae.encode(video_input).latent_dist
            latent = latent_dist.sample()  # [1, 128, T_lat, H_lat, W_lat]

        # Save latent
        latent_cpu = latent.squeeze(0).cpu().float()  # [128, T_lat, H_lat, W_lat]
        v_file = f"latents/video_{idx:04d}.pt"
        torch.save(latent_cpu, os.path.join(OUTPUT_DIR, v_file))

        # Save text embedding
        caption = caption_from_path(clip_path)
        text_emb = text_embedding_from_caption(caption)
        t_file = f"latents/text_{idx:04d}.pt"
        torch.save(text_emb, os.path.join(OUTPUT_DIR, t_file))

        metadata.append({"video": v_file, "text": t_file, "caption": caption})

        # Progress
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            remaining = (len(all_clips) - idx - 1) / rate
            print(f"  [{idx+1}/{len(all_clips)}] {rate:.1f} clips/s | ETA: {remaining/60:.0f} min")

        # Free VRAM periodically
        del video_input, latent_dist, latent, latent_cpu
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        errors.append(f"OOM: {clip_path.name}")
        continue
    except Exception as e:
        errors.append(f"{clip_path.name}: {e}")
        continue

# Save metadata
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

elapsed = time.time() - t_start
print(f"\n{'=' * 60}")
print(f"  ENCODING COMPLETE")
print(f"  Processed: {len(metadata)} clips in {elapsed/60:.1f} min")
print(f"  Errors: {len(errors)}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Latent shape: [{LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W}]")
print(f"{'=' * 60}")

if errors[:5]:
    print("\nFirst errors:")
    for e in errors[:5]:
        print(f"  {e}")
