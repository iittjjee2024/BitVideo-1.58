"""Generate the Kaggle full-scale training notebook as JSON."""
import json
import os

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {"trusted": True}, "outputs": [],
                  "source": source.split("\n")})

# --- CELLS ---

md("# BitVideo-1.58 Full-Scale Training on Kaggle\n## W1.58A8 Ternary Video Diffusion Transformer\n\n| Feature | Details |\n|---|---|\n| Model | BitVideo (dim=512, depth=8, ~50M params) |\n| Weights | Ternary {-1, 0, +1} = 1.58 bits |\n| Activations | INT8 per-token |\n| Text Encoder | Flan-T5-Base (768D) |\n| VAE | LTX-Video (128 latent channels) |\n| Training | BF16 mixed precision, 8-bit AdamW |\n| GPU | Kaggle T4 x2 (30GB) |\n\n### Setup\n1. Settings -> Accelerator -> GPU T4 x2\n2. Turn on Internet\n3. Run All")

code("# Step 1: Environment\nimport os, gc, sys, time, math\nimport torch\nimport numpy as np\nprint(f'PyTorch: {torch.__version__}')\nprint(f'GPU: {torch.cuda.get_device_name(0)}')\nprint(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\nos.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'")

md("## Step 2: Install Dependencies")

code("!pip install -q diffusers transformers accelerate safetensors sentencepiece\n!pip install -q opencv-python-headless imageio[ffmpeg] bitsandbytes\nprint('Done!')")

md("## Step 3: Upload BitVideo Package\n\nUpload your `bitvideo/` package folder as a Kaggle Dataset, or paste the key modules inline.")

code("# Option A: Clone from GitHub (if you push your repo)\n# !git clone https://github.com/YOUR_USER/bitvideo-158.git\n# sys.path.insert(0, 'bitvideo-158')\n\n# Option B: Inline minimal BitVideo for training\n# (The full package has 65 files - for Kaggle, we use the core modules)\n\n# For now, let's define the minimal training components inline:\nfrom torch.cuda.amp import GradScaler, autocast\nimport torch.nn as nn\nimport torch.nn.functional as F\nprint('Ready')")

md("## Step 4: Load Real Text Encoder")

code("from transformers import T5EncoderModel, T5Tokenizer\n\nprint('Loading Flan-T5-Base...')\ntokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')\ntext_encoder = T5EncoderModel.from_pretrained(\n    'google/flan-t5-base', torch_dtype=torch.float16\n).cuda().eval()\nfor p in text_encoder.parameters():\n    p.requires_grad = False\n\nTEXT_DIM = text_encoder.config.d_model\nprint(f'Text encoder: {TEXT_DIM}D')\n\ndef encode_text(prompts, max_length=77):\n    tokens = tokenizer(prompts, return_tensors='pt', padding='max_length',\n                       truncation=True, max_length=max_length).to('cuda')\n    with torch.no_grad():\n        return text_encoder(**tokens).last_hidden_state.float()\n\ntest = encode_text(['A dragon flying over mountains'])\nprint(f'Test: {tuple(test.shape)}')\ndel test; torch.cuda.empty_cache()")

md("## Step 5: Load LTX Video VAE")

code("from diffusers.models import AutoencoderKLLTXVideo\n\nprint('Loading LTX Video VAE...')\nvae = AutoencoderKLLTXVideo.from_pretrained(\n    'Lightricks/LTX-Video', subfolder='vae', torch_dtype=torch.float16\n).cuda().eval()\nfor p in vae.parameters():\n    p.requires_grad = False\nprint(f'VAE: {sum(p.numel() for p in vae.parameters()):,} params')\n\n# Verify\nwith torch.no_grad():\n    x = torch.randn(1, 3, 17, 128, 128, device='cuda', dtype=torch.float16)\n    z = vae.encode(x).latent_dist.sample()\n    y = vae.decode(z).sample\nprint(f'Encode: {tuple(x.shape)} -> {tuple(z.shape)}')\nprint(f'Decode: {tuple(z.shape)} -> {tuple(y.shape)}')\nLATENT_C = z.shape[1]\ndel x, z, y; torch.cuda.empty_cache()")

md("## Step 6: Prepare Video Dataset\n\nUpload your video clips as a Kaggle Dataset.\nThen encode them here with the VAE + text encoder.")

code("import cv2\nfrom pathlib import Path\n\n# Change this to your Kaggle dataset path\nVIDEO_DIR = '/kaggle/input/your-video-dataset'\nENCODED_DIR = '/kaggle/working/encoded'\nos.makedirs(f'{ENCODED_DIR}/latents', exist_ok=True)\n\nTARGET_FRAMES = 17\nTARGET_H, TARGET_W = 128, 128\n\ndef encode_clip(video_path, caption):\n    cap = cv2.VideoCapture(str(video_path))\n    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n    if total <= 0: cap.release(); return None, None\n    indices = torch.linspace(0, total-1, TARGET_FRAMES).long().tolist()\n    frames = []\n    for idx in indices:\n        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n        ret, frame = cap.read()\n        if ret:\n            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n            frame = cv2.resize(frame, (TARGET_W, TARGET_H))\n            frames.append(torch.from_numpy(frame).float() / 127.5 - 1.0)\n        elif frames: frames.append(frames[-1].clone())\n    cap.release()\n    if len(frames) < TARGET_FRAMES: return None, None\n    video = torch.stack(frames).permute(3,0,1,2).unsqueeze(0).cuda().half()\n    with torch.no_grad():\n        latent = vae.encode(video).latent_dist.sample().squeeze(0).cpu().float()\n        text_emb = encode_text([caption]).squeeze(0).cpu()\n    del video; torch.cuda.empty_cache()\n    return latent, text_emb\n\n# Encode all clips\nif Path(VIDEO_DIR).exists():\n    clips = sorted(Path(VIDEO_DIR).rglob('*.mp4'))\n    print(f'Found {len(clips)} clips')\n    metadata = []\n    for i, clip in enumerate(clips):\n        caption = clip.stem.replace('_', ' ')\n        latent, text_emb = encode_clip(clip, caption)\n        if latent is not None:\n            torch.save(latent, f'{ENCODED_DIR}/latents/video_{i:04d}.pt')\n            torch.save(text_emb, f'{ENCODED_DIR}/latents/text_{i:04d}.pt')\n            metadata.append({'video': f'latents/video_{i:04d}.pt', 'text': f'latents/text_{i:04d}.pt'})\n        if (i+1) % 50 == 0: print(f'  {i+1}/{len(clips)}')\n    import json\n    with open(f'{ENCODED_DIR}/metadata.json', 'w') as f:\n        json.dump(metadata, f)\n    print(f'Encoded {len(metadata)} clips')\nelse:\n    print(f'No dataset at {VIDEO_DIR}')\n    print('Upload your clips as a Kaggle Dataset first!')")

md("## Step 7: Define BitVideo Model (Full Scale)")

code("# Full-scale BitVideo for Kaggle T4x2\nfrom bitvideo.models import VideoDiT\n\nmodel = VideoDiT(\n    in_channels=LATENT_C,  # 128 (LTX VAE)\n    dim=512,               # 512 for T4x2 (or 768 if fits)\n    depth=8,               # 8 blocks\n    num_heads=8,\n    context_dim=TEXT_DIM,  # 768 (T5)\n    patch_size=(1, 2, 2),\n    ffn_expansion_ratio=4.0,\n    qk_norm=True,\n    device='cuda',\n    dtype=torch.float32,\n).train()\n\nprint(f'Model: {model.parameter_count():,} params')\nprint(f'Ternary weight memory: {model.parameter_count() * 2 / 8 / 1e6:.1f} MB')\nprint(f'FP16 weight memory: {model.parameter_count() * 2 / 1e6:.1f} MB')\nprint(f'Compression: 16x')")

md("## Step 8: Train!")

code("# Training config\nMAX_STEPS = 50000\nBATCH_SIZE = 1\nGRAD_ACCUM = 8\nLR = 2e-4\nWARMUP = 2000\n\n# 8-bit optimizer\ntry:\n    import bitsandbytes as bnb\n    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)\n    print('8-bit AdamW')\nexcept:\n    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)\n\ndef cosine_lr(step):\n    if step < WARMUP: return step / WARMUP\n    return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (MAX_STEPS - WARMUP)))\nlr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)\n\nfrom bitvideo.training.losses import DiffusionLoss\nfrom bitvideo.pipeline.schedulers import DDIMScheduler\nnoise_sched = DDIMScheduler(num_train_steps=1000, prediction_type='epsilon')\nloss_fn = DiffusionLoss(prediction_type='epsilon', snr_gamma=5.0)\nscaler = GradScaler()\n\nprint(f'Training: {MAX_STEPS} steps, eff. batch {BATCH_SIZE*GRAD_ACCUM}, LR {LR}')")

code("# Main training loop\nfrom torch.utils.data import Dataset, DataLoader\nimport json\n\nclass EncodedDataset(Dataset):\n    def __init__(self, root):\n        self.root = Path(root)\n        with open(self.root / 'metadata.json', 'r') as f:\n            self.samples = json.load(f)\n    def __len__(self): return len(self.samples)\n    def __getitem__(self, idx):\n        s = self.samples[idx]\n        v = torch.load(self.root / s['video'], weights_only=True)\n        t = torch.load(self.root / s['text'], weights_only=True)\n        return {'video_latent': v, 'text_embedding': t}\n\ndataset = EncodedDataset(ENCODED_DIR)\ndl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)\nprint(f'Dataset: {len(dataset)} samples')\n\nmodel.train()\nglobal_step = 0\nrunning_loss = 0.0\nt0 = time.time()\ndata_iter = iter(dl)\n\nfor step in range(MAX_STEPS * GRAD_ACCUM):\n    try: batch = next(data_iter)\n    except StopIteration: data_iter = iter(dl); batch = next(data_iter)\n    \n    video = batch['video_latent'].cuda()\n    text = batch['text_embedding'].cuda()\n    B = video.shape[0]\n    t_step = torch.randint(0, 1000, (B,), device='cuda')\n    noise = torch.randn_like(video)\n    noisy = noise_sched.add_noise(video, noise, t_step)\n    \n    with autocast(device_type='cuda', dtype=torch.bfloat16):\n        pred = model(noisy, t_step.float(), text)\n        loss = loss_fn(pred, noise, timesteps=t_step,\n                      alphas_cumprod=noise_sched.alphas_cumprod.cuda()) / GRAD_ACCUM\n    \n    scaler.scale(loss).backward()\n    running_loss += loss.item() * GRAD_ACCUM\n    \n    if (step + 1) % GRAD_ACCUM == 0:\n        scaler.unscale_(optimizer)\n        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n        scaler.step(optimizer)\n        scaler.update()\n        optimizer.zero_grad(set_to_none=True)\n        lr_sched.step()\n        global_step += 1\n        \n        if global_step % 100 == 0:\n            avg = running_loss / 100\n            elapsed = time.time() - t0\n            print(f'Step {global_step} | Loss {avg:.4f} | LR {optimizer.param_groups[0][\"lr\"]:.2e} | {elapsed:.0f}s')\n            running_loss = 0.0\n        \n        if global_step % 5000 == 0:\n            torch.save({'model_state_dict': model.state_dict(), 'step': global_step},\n                      f'/kaggle/working/bitvideo_{global_step}.pt')\n            print(f'  Saved checkpoint')\n\nprint(f'Training complete! Step {global_step}')")

md("## Step 9: Generate Real Video")

code("model.eval()\nmodel.pack_weights()\n\nprompt = 'A futuristic underwater city with bioluminescent creatures, cinematic 4K'\ntext_emb = encode_text([prompt])\n\nscheduler = DPMPlusPlusScheduler(num_train_steps=1000, prediction_type='epsilon')\npipe = BitVideoPipeline(model, scheduler, decoder=None)\n\nwith torch.no_grad():\n    latents = pipe(text_emb, num_frames=3, height=4, width=4,\n                   num_inference_steps=25, guidance_scale=1.0, decode=False)\n    decoded = vae.decode(latents.half()).sample\n\nvideo = decoded[0].permute(1,2,3,0).cpu().float().numpy()\nvideo = ((video - video.min()) / (video.max() - video.min()) * 255).clip(0,255).astype(np.uint8)\n\nimport imageio\nwriter = imageio.get_writer('/kaggle/working/output.mp4', fps=24)\nfor frame in video:\n    writer.append_data(frame)\nwriter.close()\nprint(f'Generated: {video.shape[0]} frames @ {video.shape[2]}x{video.shape[1]}')\nprint('Saved: /kaggle/working/output.mp4')")

md("---\n## Summary\n\nThis notebook trains BitVideo-1.58 at scale:\n- **Real text encoder** (Flan-T5) for prompt understanding\n- **Real VAE** (LTX-Video) for pixel-quality output\n- **128-channel latents** matching the VAE's native space\n- **50K steps** with 8-bit optimizer and BF16 compute\n\nFor production quality, train for 200K+ steps with more data.")

# Save
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": True}
    },
    "nbformat": 4, "nbformat_minor": 4, "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebooks", "bitvideo_kaggle_full_scale.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created: {out_path}")
