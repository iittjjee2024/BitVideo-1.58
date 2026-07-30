"""Demo: LTX-2.3 Distilled vs BitVideo-1.58 — Side-by-side video generation.

Requirements:
    pip install -U git+https://github.com/huggingface/diffusers
    pip install transformers accelerate safetensors

This script:
1. Generates a video with LTX-2.3 Distilled (8 steps, full precision)
2. Generates a video latent with BitVideo-1.58 (25 steps, ternary W1.58)
3. Compares speed, memory, and output statistics
"""

import os
import sys
import time
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = "A futuristic city with flying cars above the clouds, cinematic sci-fi, 4K"
OUTPUT_DIR = "outputs/demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("  LTX-2.3 Distilled vs BitVideo-1.58 Demo")
print("=" * 70)
print(f"  Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Prompt: {PROMPT}")
print("=" * 70)


# ============================================================
# PART 1: LTX-2.3 Distilled
# ============================================================
print("\n[1/2] LTX-2.3 Distilled (8 steps, BF16)")
print("-" * 50)

try:
    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()

    pipe_ltx = LTX2Pipeline.from_pretrained(
        "diffusers/LTX-2.3-Distilled-Diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe_ltx.enable_model_cpu_offload()

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Generate video
    t1 = time.time()
    video, audio = pipe_ltx(
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=512,
        height=320,
        num_frames=41,  # 8k+1 where k=5
        frame_rate=24.0,
        num_inference_steps=8,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        output_type="np",
        return_dict=False,
    )
    gen_time = time.time() - t1
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

    # Save video
    from diffusers.pipelines.ltx2.export_utils import encode_video
    output_path = os.path.join(OUTPUT_DIR, "ltx23_distilled.mp4")
    encode_video(
        video[0],
        fps=24.0,
        audio=audio[0].float().cpu() if audio is not None else None,
        audio_sample_rate=pipe_ltx.vocoder.config.output_sampling_rate if hasattr(pipe_ltx, 'vocoder') else 16000,
        output_path=output_path,
    )

    ltx_results = {
        "time": gen_time,
        "peak_mem_gb": peak_mem,
        "frames": video[0].shape[0],
        "resolution": f"{video[0].shape[2]}x{video[0].shape[1]}",
        "steps": 8,
        "output": output_path,
    }
    print(f"  Generated: {ltx_results['frames']} frames @ {ltx_results['resolution']}")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Peak memory: {peak_mem:.2f} GB")
    print(f"  Saved: {output_path}")

    # Free memory
    del pipe_ltx
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None

except ImportError as e:
    print(f"  SKIPPED: diffusers not installed or LTX2 not available")
    print(f"  Install: pip install -U git+https://github.com/huggingface/diffusers")
    print(f"  Error: {e}")
    ltx_results = None

except Exception as e:
    print(f"  ERROR: {e}")
    ltx_results = None
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None


# ============================================================
# PART 2: BitVideo-1.58 (Your trained model)
# ============================================================
print("\n[2/2] BitVideo-1.58 (25 steps, W1.58A8 Ternary)")
print("-" * 50)

from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DPMPlusPlusScheduler

torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
t0 = time.time()

# Load trained model
model = VideoDiT(
    in_channels=4, dim=128, depth=4, num_heads=4,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
    device=device, dtype=torch.float32,
)

ckpt_path = "outputs/checkpoint-5000.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint: step {ckpt['global_step']}")
else:
    print(f"  No checkpoint found, using random weights")

model.eval()
model.pack_weights()
load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s")
print(f"  Parameters: {model.parameter_count():,} (all ternary W1.58)")

# Generate
scheduler = DPMPlusPlusScheduler(num_train_steps=1000, prediction_type="epsilon")
pipe_bv = BitVideoPipeline(model, scheduler, decoder=None)

context = torch.randn(1, 77, 768, device=device, dtype=torch.float32)

t1 = time.time()
with torch.no_grad():
    latents = pipe_bv(
        context,
        num_frames=16,
        height=32,
        width=32,
        num_inference_steps=25,
        guidance_scale=1.0,
        decode=False,
    )
gen_time_bv = time.time() - t1
peak_mem_bv = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

# Save latent
output_path_bv = os.path.join(OUTPUT_DIR, "bitvideo158_latent.pt")
torch.save(latents.cpu(), output_path_bv)

bv_results = {
    "time": gen_time_bv,
    "peak_mem_gb": peak_mem_bv,
    "frames": 16,
    "resolution": "32x32 (latent)",
    "steps": 25,
    "output": output_path_bv,
    "params": model.parameter_count(),
    "latent_stats": {
        "mean": latents.mean().item(),
        "std": latents.std().item(),
    },
}
print(f"  Generated: {bv_results['frames']} frames @ {bv_results['resolution']}")
print(f"  Time: {gen_time_bv:.1f}s")
print(f"  Peak memory: {peak_mem_bv:.3f} GB")
print(f"  Latent mean={bv_results['latent_stats']['mean']:.4f}, std={bv_results['latent_stats']['std']:.4f}")
print(f"  Saved: {output_path_bv}")


# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("  COMPARISON")
print("=" * 70)
print(f"{'Metric':<25} {'LTX-2.3 Distilled':<25} {'BitVideo-1.58':<25}")
print("-" * 70)

if ltx_results:
    print(f"{'Steps':<25} {ltx_results['steps']:<25} {bv_results['steps']:<25}")
    print(f"{'Gen Time':<25} {ltx_results['time']:.1f}s{'':<20} {bv_results['time']:.1f}s")
    print(f"{'Peak Memory':<25} {ltx_results['peak_mem_gb']:.2f} GB{'':<17} {bv_results['peak_mem_gb']:.3f} GB")
    print(f"{'Resolution':<25} {ltx_results['resolution']:<25} {bv_results['resolution']:<25}")
    print(f"{'Frames':<25} {ltx_results['frames']:<25} {bv_results['frames']:<25}")
    mem_ratio = ltx_results['peak_mem_gb'] / max(bv_results['peak_mem_gb'], 0.001)
    print(f"\n  BitVideo uses {mem_ratio:.0f}x LESS memory than LTX-2.3!")
else:
    print(f"{'Steps':<25} {'(skipped)':<25} {bv_results['steps']:<25}")
    print(f"{'Gen Time':<25} {'N/A':<25} {bv_results['time']:.1f}s")
    print(f"{'Peak Memory':<25} {'N/A':<25} {bv_results['peak_mem_gb']:.3f} GB")
    print(f"{'Parameters':<25} {'~22B':<25} {bv_results['params']:,}")

print(f"\n{'Weight Precision':<25} {'BF16 (16 bit)':<25} {'Ternary (1.58 bit)':<25}")
print(f"{'Weight Memory':<25} {'~44 GB (full model)':<25} {bv_results['params']*2/8/1e6:.1f} MB")
print("=" * 70)
print("\nDemo complete!")
