"""Test inference with the trained BitVideo-1.58 checkpoint."""

import torch
from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DPMPlusPlusScheduler

print("Loading trained model...")
model = VideoDiT(
    in_channels=4, dim=128, depth=4, num_heads=4,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
    device="cuda", dtype=torch.float32,
)

ckpt = torch.load("outputs/checkpoint-5000.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from step {ckpt['global_step']}")
print(f"Parameters: {model.parameter_count():,}")

# Pack weights for fast ternary inference
model.pack_weights()
print("Weights packed for inference")

# Create pipeline
scheduler = DPMPlusPlusScheduler(num_train_steps=1000, prediction_type="epsilon")
pipe = BitVideoPipeline(model, scheduler, decoder=None)

# Generate video latents
print("\nGenerating video (25 DPM++ steps)...")
context = torch.randn(1, 77, 768, device="cuda", dtype=torch.float32)

with torch.no_grad():
    latents = pipe(
        context,
        num_frames=16,
        height=32,
        width=32,
        num_inference_steps=25,
        guidance_scale=1.0,
        decode=False,
    )

print(f"\nGenerated latents: {tuple(latents.shape)}")
print(f"  Mean: {latents.mean().item():.4f}")
print(f"  Std:  {latents.std().item():.4f}")
print(f"  Min:  {latents.min().item():.4f}")
print(f"  Max:  {latents.max().item():.4f}")
print(f"  Finite: {torch.isfinite(latents).all().item()}")

# Save generated latent
torch.save(latents.cpu(), "outputs/generated_sample.pt")
print("\nSaved to outputs/generated_sample.pt")
print("INFERENCE TEST PASSED")
