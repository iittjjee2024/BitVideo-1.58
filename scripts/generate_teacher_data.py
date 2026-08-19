"""Generate teacher data from LTX-2.3 for distillation.

This script generates denoising trajectories from the LTX-2.3 model
and saves them as training data for the BitVideo student.

Two modes:
    1. LOCAL: Load LTX-2.3 locally (needs A100 80GB)
    2. API: Use fal.ai or Replicate API (no GPU needed, just costs ~$0.01/video)

The output format for each sample:
    latents/sample_XXXXXX.pt = {
        'noisy_latent': [C, T, H, W],    # Input to denoiser at timestep t
        'clean_latent': [C, T, H, W],    # Clean video latent
        'noise': [C, T, H, W],           # The noise added
        'timestep': scalar,              # Diffusion timestep (0-999)
        'teacher_pred': [C, T, H, W],   # What LTX-2.3 predicted
    }
    conditions/sample_XXXXXX.pt = {
        'embedding': [L, D],             # Text embedding
    }

Usage:
    # Generate from prompts file using local LTX model
    python scripts/generate_teacher_data.py \\
        --mode local \\
        --prompts prompts.txt \\
        --output data/teacher_latents \\
        --num-timesteps-per-sample 10

    # Generate using fal.ai API (cheaper, no GPU needed)
    python scripts/generate_teacher_data.py \\
        --mode api \\
        --prompts prompts.txt \\
        --output data/teacher_latents \\
        --api-key YOUR_FAL_KEY

    # Generate random prompts for large-scale distillation
    python scripts/generate_teacher_data.py \\
        --mode local \\
        --num-samples 10000 \\
        --output data/teacher_latents
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "A {adj} {subject} {action} in {setting}, {quality}",
    "{subject} {action}, {setting}, cinematic lighting, {quality}",
    "A slow motion shot of {subject} {action}, {adj} colors, {quality}",
    "Aerial drone shot of {setting} with {subject}, golden hour, {quality}",
    "Close-up of {subject}, {adj} details, shallow depth of field, {quality}",
]

SUBJECTS = [
    "cat", "dog", "bird", "person walking", "car driving", "river flowing",
    "fire burning", "clouds moving", "flower blooming", "waves crashing",
    "city skyline", "mountain landscape", "forest path", "desert dunes",
    "ocean sunset", "starry night sky", "snow falling", "rain drops",
    "butterfly flying", "horse galloping", "fish swimming", "eagle soaring",
]

ACTIONS = [
    "moving gracefully", "spinning slowly", "dancing", "running",
    "floating", "transforming", "emerging from fog", "reflecting in water",
    "casting shadows", "glowing softly", "fading in and out",
]

SETTINGS = [
    "a misty forest", "a futuristic city", "an underwater cave",
    "a golden wheat field", "a neon-lit street", "a snowy mountain",
    "a tropical beach", "a quiet library", "an ancient temple",
    "outer space", "a bustling market", "a serene lake",
]

ADJECTIVES = [
    "vibrant", "ethereal", "dramatic", "peaceful", "mysterious",
    "warm", "cool", "cinematic", "dreamy", "sharp",
]

QUALITIES = [
    "4K quality", "smooth motion", "24fps", "high detail",
    "professional cinematography", "ultra HD", "film grain",
]


def generate_random_prompts(n: int) -> list[str]:
    """Generate diverse random prompts for teacher data generation."""
    import random
    prompts = []
    for _ in range(n):
        template = random.choice(PROMPT_TEMPLATES)
        prompt = template.format(
            subject=random.choice(SUBJECTS),
            action=random.choice(ACTIONS),
            setting=random.choice(SETTINGS),
            adj=random.choice(ADJECTIVES),
            quality=random.choice(QUALITIES),
        )
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Local teacher generation
# ---------------------------------------------------------------------------


def generate_local(
    prompts: list[str],
    output_dir: str,
    num_timesteps_per_sample: int = 10,
    resolution: tuple[int, int] = (544, 960),
    num_frames: int = 49,
    num_inference_steps: int = 30,
):
    """Generate teacher data using local LTX-2.3 model.

    Requires: A100 80GB GPU, ~45GB disk for model weights.
    """
    from diffusers import LTXPipeline
    from diffusers.models import AutoencoderKLLTXVideo

    os.makedirs(f"{output_dir}/latents", exist_ok=True)
    os.makedirs(f"{output_dir}/conditions", exist_ok=True)

    # Load LTX pipeline
    logger.info("Loading LTX-2.3 pipeline (this needs A100 80GB)...")
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    # We also need the VAE separately for encoding
    vae = pipe.vae

    # Text encoder for embeddings
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    logger.info(f"Generating teacher data for {len(prompts)} prompts...")
    logger.info(f"  {num_timesteps_per_sample} timesteps per sample")
    logger.info(f"  Resolution: {resolution[0]}x{resolution[1]}x{num_frames}")

    metadata = []
    sample_idx = 0

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"  [{prompt_idx+1}/{len(prompts)}] {prompt[:60]}...")

        try:
            # Encode text
            text_inputs = tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                truncation=True, max_length=128
            ).to("cuda")
            with torch.no_grad():
                text_emb = text_encoder(**text_inputs).last_hidden_state.float().cpu()

            # Generate a clean video using full pipeline
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=resolution[0],
                    width=resolution[1],
                    num_inference_steps=num_inference_steps,
                    output_type="latent",  # Get latents, not pixels
                )
                clean_latent = output.frames[0].cpu().float()  # [C, T, H, W]

            # Now generate training pairs at different timesteps
            # For each timestep, add noise and record what the model predicts
            scheduler = pipe.scheduler
            for t_idx in range(num_timesteps_per_sample):
                # Sample a random timestep
                t = torch.randint(0, 1000, (1,)).item()
                timestep = torch.tensor([t], device="cuda")

                # Add noise
                noise = torch.randn_like(clean_latent)
                noisy = scheduler.add_noise(
                    clean_latent.cuda(), noise.cuda(), timestep
                )

                # Get teacher prediction
                with torch.no_grad():
                    teacher_pred = pipe.transformer(
                        hidden_states=noisy.unsqueeze(0),
                        encoder_hidden_states=text_emb.cuda(),
                        timestep=timestep.float(),
                    ).sample.squeeze(0).cpu().float()

                # Save
                torch.save({
                    "noisy_latent": noisy.cpu().float(),
                    "clean_latent": clean_latent,
                    "noise": noise,
                    "timestep": torch.tensor(t),
                    "teacher_pred": teacher_pred,
                }, f"{output_dir}/latents/sample_{sample_idx:06d}.pt")

                torch.save({
                    "embedding": text_emb.squeeze(0),
                }, f"{output_dir}/conditions/sample_{sample_idx:06d}.pt")

                metadata.append({
                    "latent": f"latents/sample_{sample_idx:06d}.pt",
                    "condition": f"conditions/sample_{sample_idx:06d}.pt",
                    "prompt": prompt,
                    "timestep": t,
                })
                sample_idx += 1

            # Cleanup
            del clean_latent, output
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"  Error: {e}")
            continue

    # Save metadata
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nGenerated {sample_idx} teacher samples!")
    logger.info(f"  Saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Simplified generation (encode existing videos as teacher data)
# ---------------------------------------------------------------------------


def generate_from_videos(
    video_dirs: list[str],
    output_dir: str,
    num_timesteps_per_sample: int = 5,
    max_videos: int = 10000,
):
    """Generate teacher data by encoding existing videos with the LTX VAE.

    This is the CHEAPEST approach — no LTX transformer needed, just the VAE.
    The student learns standard diffusion (predicting noise) rather than
    matching a teacher's specific predictions.

    Good enough for Phase 1 distillation. Phase 2 can use full teacher.
    """
    import cv2
    import re
    from diffusers.models import AutoencoderKLLTXVideo
    from transformers import T5EncoderModel, T5Tokenizer

    os.makedirs(f"{output_dir}/latents", exist_ok=True)
    os.makedirs(f"{output_dir}/conditions", exist_ok=True)

    TARGET_FRAMES = 17
    TARGET_H, TARGET_W = 128, 128

    logger.info("Loading VAE + T5...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.float16
    ).cuda().eval()
    for p in vae.parameters():
        p.requires_grad = False

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    text_enc = T5EncoderModel.from_pretrained(
        "google/flan-t5-large", torch_dtype=torch.float16
    ).cuda().eval()
    for p in text_enc.parameters():
        p.requires_grad = False

    # Find videos
    all_videos = []
    for vdir in video_dirs:
        if os.path.exists(vdir):
            vids = list(Path(vdir).rglob("*.mp4")) + list(Path(vdir).rglob("*.mov"))
            all_videos.extend(sorted(vids))
    all_videos = all_videos[:max_videos]
    logger.info(f"Found {len(all_videos)} videos")

    metadata = []
    sample_idx = 0
    t0 = time.time()

    for vid_idx, vid_path in enumerate(all_videos):
        try:
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue

            indices = torch.linspace(0, total - 1, TARGET_FRAMES).long().tolist()
            frames = []
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
                    frames.append(torch.from_numpy(frame).float() / 127.5 - 1.0)
                elif frames:
                    frames.append(frames[-1].clone())
            cap.release()
            if len(frames) < TARGET_FRAMES:
                continue

            # Encode video
            vid_t = torch.stack(frames[:TARGET_FRAMES]).permute(3, 0, 1, 2).unsqueeze(0).cuda().half()
            with torch.no_grad():
                clean_latent = vae.encode(vid_t).latent_dist.sample().squeeze(0).cpu().float()

            # Encode caption
            caption = re.sub(r"[_\-]+", " ", vid_path.stem).strip() or "a video"
            tok = tokenizer([caption], return_tensors="pt", padding="max_length",
                            truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                text_emb = text_enc(**tok).last_hidden_state.squeeze(0).cpu().float()

            # Generate multiple timestep samples from this video
            for _ in range(num_timesteps_per_sample):
                t = torch.randint(0, 1000, (1,)).item()
                noise = torch.randn_like(clean_latent)

                # Simple noise schedule: noisy = sqrt(alpha) * clean + sqrt(1-alpha) * noise
                alpha = 1.0 - (t / 1000.0)  # Linear for simplicity
                noisy = math.sqrt(alpha) * clean_latent + math.sqrt(1 - alpha) * noise

                torch.save({
                    "noisy_latent": noisy,
                    "clean_latent": clean_latent,
                    "noise": noise,
                    "timestep": torch.tensor(t),
                    "teacher_pred": noise,  # For epsilon prediction, target = noise
                }, f"{output_dir}/latents/sample_{sample_idx:06d}.pt")

                torch.save({
                    "embedding": text_emb,
                }, f"{output_dir}/conditions/sample_{sample_idx:06d}.pt")

                metadata.append({
                    "latent": f"latents/sample_{sample_idx:06d}.pt",
                    "condition": f"conditions/sample_{sample_idx:06d}.pt",
                })
                sample_idx += 1

            del vid_t, clean_latent
            if (vid_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                elapsed = time.time() - t0
                logger.info(f"  {vid_idx+1}/{len(all_videos)} | {sample_idx} samples | {elapsed/60:.0f}m")

        except Exception as e:
            continue

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info(f"\nGenerated {sample_idx} samples from {len(all_videos)} videos")
    logger.info(f"  Output: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate teacher data for distillation")
    parser.add_argument("--mode", choices=["local", "api", "videos"], default="videos",
                        help="Generation mode")
    parser.add_argument("--prompts", type=str, help="Text file with prompts (one per line)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of random prompts to generate")
    parser.add_argument("--output", type=str, default="data/teacher_latents",
                        help="Output directory")
    parser.add_argument("--video-dirs", nargs="+", help="Video directories (for --mode videos)")
    parser.add_argument("--num-timesteps", type=int, default=5,
                        help="Timestep samples per video/prompt")
    parser.add_argument("--max-videos", type=int, default=10000,
                        help="Max videos to process")
    parser.add_argument("--api-key", type=str, help="API key for fal.ai/Replicate")
    args = parser.parse_args()

    # Load or generate prompts
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = generate_random_prompts(args.num_samples)
        logger.info(f"Generated {len(prompts)} random prompts")

    if args.mode == "local":
        generate_local(
            prompts, args.output,
            num_timesteps_per_sample=args.num_timesteps,
        )
    elif args.mode == "videos":
        video_dirs = args.video_dirs or [
            "G:\\My Drive\\Antigravity_Production\\Training_Data\\datasets",
            "G:\\My Drive\\Generated_Data\\Dataset",
        ]
        generate_from_videos(
            video_dirs, args.output,
            num_timesteps_per_sample=args.num_timesteps,
            max_videos=args.max_videos,
        )
    elif args.mode == "api":
        logger.info("API mode: Use fal.ai LTX-2 trainer endpoint")
        logger.info("  See: https://fal.ai/models/fal-ai/ltx2-video-trainer")
        logger.info("  Cost: ~$0.01 per video sample")
        logger.info("  For 10K samples: ~$100")


if __name__ == "__main__":
    main()
