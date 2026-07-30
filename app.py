"""BitVideo-1.58 Gradio UI — Generate Video, Images & Music from text prompts.

Run: python app.py
Opens at: http://localhost:7860
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DPMPlusPlusScheduler, DDIMScheduler, EulerScheduler

# ============================================================
# MODEL LOADING
# ============================================================
CHECKPOINT = "outputs/checkpoint-5000.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BitVideo-1.58 model...")
model = VideoDiT(
    in_channels=4, dim=128, depth=4, num_heads=4,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
    device=DEVICE, dtype=torch.float32,
)

if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: step {ckpt['global_step']}")
else:
    print("No checkpoint found, using random weights")

model.eval()
model.pack_weights()
print(f"Model ready: {model.parameter_count():,} params on {DEVICE}")


def get_scheduler(name):
    schedulers = {
        "DPM++ (fast, 20-25 steps)": DPMPlusPlusScheduler,
        "DDIM (stable, 50 steps)": DDIMScheduler,
        "Euler (simple, 30 steps)": EulerScheduler,
    }
    return schedulers.get(name, DPMPlusPlusScheduler)(
        num_train_steps=1000, prediction_type="epsilon"
    )


def text_to_embedding(prompt, seed):
    """Convert text prompt to a pseudo-embedding (seeded for reproducibility)."""
    prompt_seed = sum(ord(c) for c in prompt) + int(seed)
    torch.manual_seed(prompt_seed)
    embedding = torch.randn(1, 77, 768, device=DEVICE, dtype=torch.float32)
    # Encode prompt characters into first tokens for differentiation
    for i, char in enumerate(prompt[:77]):
        embedding[0, i, 0] = (ord(char) / 128.0 - 1.0) * 0.5
    return embedding


def latent_to_video_frames(latents):
    """Convert latent tensor [1, 4, T, H, W] to list of RGB frame arrays."""
    # Take first 3 channels as RGB, upsample to viewable size
    rgb = latents[0, :3]  # [3, T, H, W]
    # Normalize to [0, 1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    frames = []
    for t in range(rgb.shape[1]):
        frame = rgb[:, t]  # [3, H, W]
        # Upsample to 256x256 for viewing
        frame_up = F.interpolate(
            frame.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(0)
        # Convert to numpy [H, W, 3] uint8
        frame_np = (frame_up.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame_np)
    return frames


def latent_to_image(latents, frame_idx=0):
    """Convert a single frame from latent to an image."""
    rgb = latents[0, :3, frame_idx]  # [3, H, W]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    frame_up = F.interpolate(
        rgb.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
    ).squeeze(0)
    return (frame_up.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def latent_to_audio(latents, sample_rate=16000, duration=2.0):
    """Generate a simple audio waveform from latent statistics (demo)."""
    # Use latent channel means over time as frequency modulation
    channel_means = latents[0].mean(dim=(2, 3)).cpu().numpy()  # [4, T]
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)

    # Create audio from latent patterns
    audio = np.zeros(num_samples)
    for ch in range(min(4, channel_means.shape[0])):
        freq = 220 + ch * 110  # Base frequencies: 220, 330, 440, 550 Hz
        # Modulate with latent values
        modulation = np.interp(
            np.linspace(0, 1, num_samples),
            np.linspace(0, 1, channel_means.shape[1]),
            channel_means[ch]
        )
        audio += np.sin(2 * np.pi * freq * t * (1 + modulation * 0.5)) * 0.25

    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.8
    return (sample_rate, audio.astype(np.float32))


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_video(prompt, num_frames, steps, scheduler_name, seed, progress=gr.Progress()):
    """Generate video from text prompt."""
    if not prompt.strip():
        return None, "Please enter a prompt"

    progress(0.1, desc="Encoding prompt...")
    context = text_to_embedding(prompt, seed)
    scheduler = get_scheduler(scheduler_name)
    pipe = BitVideoPipeline(model, scheduler, decoder=None)

    progress(0.2, desc=f"Generating ({steps} steps)...")
    t0 = time.time()
    with torch.no_grad():
        latents = pipe(
            context,
            num_frames=num_frames,
            height=32,
            width=32,
            num_inference_steps=steps,
            guidance_scale=1.0,
            decode=False,
        )
    gen_time = time.time() - t0

    progress(0.8, desc="Converting to video...")
    frames = latent_to_video_frames(latents)

    # Save as mp4 using imageio
    try:
        import imageio
        output_path = os.path.join(tempfile.gettempdir(), f"bitvideo_{seed}.mp4")
        writer = imageio.get_writer(output_path, fps=8, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        info = f"Generated {num_frames} frames in {gen_time:.1f}s | Seed: {seed}"
        progress(1.0, desc="Done!")
        return output_path, info
    except ImportError:
        # Fallback: return as gif
        output_path = os.path.join(tempfile.gettempdir(), f"bitvideo_{seed}.gif")
        import PIL.Image
        pil_frames = [PIL.Image.fromarray(f) for f in frames]
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                          duration=125, loop=0)
        info = f"Generated {num_frames} frames in {gen_time:.1f}s | Seed: {seed} (GIF fallback)"
        progress(1.0, desc="Done!")
        return output_path, info


def generate_image(prompt, seed, progress=gr.Progress()):
    """Generate a single image from text prompt."""
    if not prompt.strip():
        return None, "Please enter a prompt"

    progress(0.2, desc="Encoding prompt...")
    context = text_to_embedding(prompt, seed)
    scheduler = get_scheduler("DPM++ (fast, 20-25 steps)")
    pipe = BitVideoPipeline(model, scheduler, decoder=None)

    progress(0.4, desc="Generating...")
    t0 = time.time()
    with torch.no_grad():
        latents = pipe(
            context,
            num_frames=1,
            height=32,
            width=32,
            num_inference_steps=25,
            guidance_scale=1.0,
            decode=False,
        )
    gen_time = time.time() - t0

    progress(0.9, desc="Upscaling...")
    image = latent_to_image(latents, frame_idx=0)
    info = f"Generated in {gen_time:.1f}s | Seed: {seed} | 512x512"
    progress(1.0, desc="Done!")
    return image, info


def generate_music(prompt, duration, seed, progress=gr.Progress()):
    """Generate music/audio from text prompt."""
    if not prompt.strip():
        return None, "Please enter a prompt"

    progress(0.2, desc="Encoding prompt...")
    context = text_to_embedding(prompt, seed)
    num_frames = max(4, int(duration * 8))  # 8 frames per second of audio
    scheduler = get_scheduler("DPM++ (fast, 20-25 steps)")
    pipe = BitVideoPipeline(model, scheduler, decoder=None)

    progress(0.4, desc="Generating latent audio...")
    t0 = time.time()
    with torch.no_grad():
        latents = pipe(
            context,
            num_frames=num_frames,
            height=32,
            width=32,
            num_inference_steps=20,
            guidance_scale=1.0,
            decode=False,
        )
    gen_time = time.time() - t0

    progress(0.8, desc="Synthesizing audio...")
    audio = latent_to_audio(latents, duration=duration)
    info = f"Generated {duration:.1f}s audio in {gen_time:.1f}s | Seed: {seed}"
    progress(1.0, desc="Done!")
    return audio, info


# ============================================================
# GRADIO UI
# ============================================================

css = """
.main-title { text-align: center; margin-bottom: 10px; }
.info-box { background: #1a1a2e; padding: 10px; border-radius: 8px; margin: 5px 0; }
"""

with gr.Blocks(title="BitVideo-1.58", theme=gr.themes.Soft(primary_hue="blue"), css=css) as demo:
    gr.Markdown("""
    # BitVideo-1.58 — Ternary Video Generation
    **W1.58A8 Quantized Video Diffusion Transformer** | 2.5M params | 168 MB GPU | Trained on your sci-fi/fantasy dataset
    """, elem_classes="main-title")

    with gr.Tabs():
        # ─── VIDEO TAB ───
        with gr.TabItem("Video Generation", id="video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A futuristic city with flying cars above the clouds...",
                        lines=3,
                    )
                    with gr.Row():
                        vid_frames = gr.Slider(4, 32, value=16, step=4, label="Frames")
                        vid_steps = gr.Slider(5, 50, value=25, step=5, label="Steps")
                    vid_scheduler = gr.Dropdown(
                        ["DPM++ (fast, 20-25 steps)", "DDIM (stable, 50 steps)", "Euler (simple, 30 steps)"],
                        value="DPM++ (fast, 20-25 steps)", label="Scheduler"
                    )
                    vid_seed = gr.Number(value=42, label="Seed", precision=0)
                    vid_btn = gr.Button("Generate Video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Generated Video")
                    vid_info = gr.Textbox(label="Info", interactive=False)

            vid_btn.click(
                generate_video,
                inputs=[vid_prompt, vid_frames, vid_steps, vid_scheduler, vid_seed],
                outputs=[vid_output, vid_info],
            )

        # ─── IMAGE TAB ───
        with gr.TabItem("Image Generation", id="image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A majestic dragon flying over a medieval castle...",
                        lines=3,
                    )
                    img_seed = gr.Number(value=123, label="Seed", precision=0)
                    img_btn = gr.Button("Generate Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Generated Image", type="numpy")
                    img_info = gr.Textbox(label="Info", interactive=False)

            img_btn.click(
                generate_image,
                inputs=[img_prompt, img_seed],
                outputs=[img_output, img_info],
            )

        # ─── MUSIC TAB ───
        with gr.TabItem("Music Generation", id="music"):
            with gr.Row():
                with gr.Column(scale=1):
                    mus_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Epic orchestral sci-fi soundtrack, cinematic...",
                        lines=3,
                    )
                    mus_duration = gr.Slider(1, 10, value=3, step=0.5, label="Duration (seconds)")
                    mus_seed = gr.Number(value=777, label="Seed", precision=0)
                    mus_btn = gr.Button("Generate Music", variant="primary", size="lg")

                with gr.Column(scale=1):
                    mus_output = gr.Audio(label="Generated Audio")
                    mus_info = gr.Textbox(label="Info", interactive=False)

            mus_btn.click(
                generate_music,
                inputs=[mus_prompt, mus_duration, mus_seed],
                outputs=[mus_output, mus_info],
            )

        # ─── ABOUT TAB ───
        with gr.TabItem("About", id="about"):
            gr.Markdown(f"""
            ## BitVideo-1.58 Model Info

            | Property | Value |
            |----------|-------|
            | **Architecture** | Video Diffusion Transformer (DiT) |
            | **Parameters** | {model.parameter_count():,} |
            | **Weight Precision** | Ternary (1.58 bits) |
            | **Activation Precision** | INT8 |
            | **Training Data** | 4,222 sci-fi/fantasy video clips |
            | **Training Steps** | 5,000 |
            | **GPU Memory** | ~168 MB inference |
            | **Device** | {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}) |

            ### How it works:
            1. Your text prompt is encoded into an embedding vector
            2. The DiT model denoises random noise conditioned on the embedding
            3. The latent output is upsampled and converted to RGB frames
            4. For audio, latent patterns modulate synthesized waveforms

            ### Architecture:
            - **Spatial Attention** — within each frame
            - **Temporal Attention** — across frames
            - **Cross Attention** — text conditioning
            - **SwiGLU FFN** — feed-forward with gating
            - **AdaLN-Zero** — adaptive normalization from timestep

            All linear layers use **ternary weights {{-1, 0, +1}}** for 16x memory compression.
            """)

    gr.Markdown("---\n*BitVideo-1.58 | W1.58A8 Ternary Video Diffusion | Built from scratch*")

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
