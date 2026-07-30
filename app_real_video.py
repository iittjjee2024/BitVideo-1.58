"""BitVideo-1.58 — Real Video Generation with LTX VAE Decoder.

Generates actual video frames by:
1. BitVideo DiT generates 128-channel latents
2. LTX Video VAE decodes latents to real RGB frames
3. Output saved as MP4

Run: python app_real_video.py
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs_128ch/checkpoint-3000.pt"
VAE_CACHE = "C:/bitvideo_vae"

# ─── Load BitVideo ───
print("Loading BitVideo-1.58 (128ch)...")
from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DPMPlusPlusScheduler

model = VideoDiT(
    in_channels=128, dim=128, depth=4, num_heads=4,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
    device=DEVICE, dtype=torch.float32,
)

if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded step {ckpt['global_step']}")
else:
    # Try earlier checkpoint
    for step in [2000, 1000]:
        alt = f"outputs_128ch/checkpoint-{step}.pt"
        if os.path.exists(alt):
            ckpt = torch.load(alt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded step {ckpt['global_step']}")
            break
    else:
        print("  WARNING: No checkpoint found, using random weights")

model.eval()
model.pack_weights()
print(f"  BitVideo ready: {model.parameter_count():,} params")

# ─── Load LTX VAE ───
print("Loading LTX Video VAE...")
from diffusers.models import AutoencoderKLLTXVideo

vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video", subfolder="vae",
    torch_dtype=torch.float16, cache_dir=VAE_CACHE,
)
vae = vae.to(DEVICE).eval()
print(f"  VAE ready: {sum(p.numel() for p in vae.parameters()):,} params")
print(f"  Total GPU: ~{torch.cuda.memory_allocated()/1e9:.1f} GB")


@torch.no_grad()
def generate_real_video(prompt, steps, seed):
    """Full pipeline: text → BitVideo latent → LTX VAE decode → MP4."""
    # Encode prompt
    prompt_seed = sum(ord(c) for c in prompt) + int(seed)
    torch.manual_seed(prompt_seed)
    context = torch.randn(1, 77, 768, device=DEVICE, dtype=torch.float32)
    for i, ch in enumerate(prompt[:77]):
        context[0, i, 0] = (ord(ch) / 128.0 - 1.0) * 0.5

    # Generate 128ch latent with BitVideo
    scheduler = DPMPlusPlusScheduler(num_train_steps=1000, prediction_type="epsilon")
    pipe = BitVideoPipeline(model, scheduler, decoder=None)

    latents = pipe(
        context,
        num_frames=3,   # Temporal latent frames (VAE outputs ~17 pixel frames)
        height=4,       # Spatial latent size
        width=4,
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        decode=False,
    )
    # latents: [1, 128, 3, 4, 4]

    # Decode with LTX VAE
    latents_fp16 = latents.to(dtype=torch.float16)
    decoded = vae.decode(latents_fp16).sample  # [1, 3, T_px, H_px, W_px]

    # Convert to numpy [T, H, W, 3] uint8
    video = decoded[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video = (video - video.min()) / (video.max() - video.min() + 1e-8)
    frames = (video * 255).clip(0, 255).astype(np.uint8)

    return frames


def generate(prompt, steps, seed, progress=gr.Progress()):
    if not prompt.strip():
        return None, "Enter a prompt"

    progress(0.1, desc="Generating latents with BitVideo...")
    t0 = time.time()

    try:
        frames = generate_real_video(prompt, steps, seed)
        gen_time = time.time() - t0

        progress(0.9, desc="Saving video...")
        output_path = os.path.join(tempfile.gettempdir(), f"bitvideo_real_{int(seed)}.mp4")

        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=24, codec="libx264", quality=7)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except ImportError:
            import PIL.Image
            output_path = output_path.replace(".mp4", ".gif")
            pil_frames = [PIL.Image.fromarray(f) for f in frames]
            pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                              duration=42, loop=0)

        n_frames = frames.shape[0]
        h, w = frames.shape[1], frames.shape[2]
        info = f"{n_frames} frames @ {w}x{h} | {gen_time:.1f}s | Seed: {int(seed)}"
        progress(1.0)
        return output_path, info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}"


# ─── Gradio UI ───
print("\nLaunching...")
with gr.Blocks(title="BitVideo-1.58 Real Video") as demo:
    gr.Markdown("""
    # BitVideo-1.58 — Real Video Generation
    **Ternary W1.58 DiT → LTX Video VAE Decoder → Actual Video**

    Your trained model generates 128-channel latents that the LTX VAE decodes into real RGB video frames.
    """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=2,
                              value="A futuristic city with flying cars above the clouds, cinematic")
            steps = gr.Slider(10, 50, value=25, step=5, label="Steps")
            seed = gr.Number(value=42, label="Seed", precision=0)
            btn = gr.Button("Generate Video", variant="primary", size="lg")
        with gr.Column():
            video_out = gr.Video(label="Generated Video")
            info_out = gr.Textbox(label="Info", interactive=False)

    btn.click(generate, [prompt, steps, seed], [video_out, info_out])

    gr.Markdown(f"---\n**BitVideo**: {model.parameter_count():,} params (ternary) | **VAE**: 419M params | **GPU**: ~2.5 GB")

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
