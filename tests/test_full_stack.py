"""Comprehensive integration test for the full BitVideo-1.58 stack.

Validates every major component: quantization, model primitives, attention
variants, transformer, pipeline, training, and extras.
"""

import math
import torch
import pytest

# ─── Quantization ───
def test_ternary_quantization():
    from bitvideo.quantization import BitLinear
    linear = BitLinear(32, 16).eval()
    x = torch.randn(4, 32)
    with torch.no_grad():
        out = linear(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()

def test_bitlinear_gradient():
    from bitvideo.quantization import BitLinear
    linear = BitLinear(32, 16)
    x = torch.randn(4, 32, requires_grad=True)
    linear(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None

# ─── Model Primitives ───
def test_rope():
    from bitvideo.models import RotaryEmbedding, apply_rotary_embedding
    rope = RotaryEmbedding(8)
    x = torch.randn(2, 3, 8)
    out = apply_rotary_embedding(x, rope(3))
    assert out.shape == x.shape
    torch.testing.assert_close(out.square().sum(-1), x.square().sum(-1), rtol=1e-5, atol=1e-6)

def test_patch_embed():
    from bitvideo.models import VideoPatchEmbed, patchify_video, unpatchify_video
    video = torch.randn(1, 3, 2, 6, 6)
    patches, info = patchify_video(video, (1, 2, 2))
    restored = unpatchify_video(patches, info, channels=3)
    torch.testing.assert_close(restored, video, rtol=0, atol=0)

def test_feedforward():
    from bitvideo.models import FeedForward
    ffn = FeedForward(16, hidden_features=32, activation="swiglu")
    x = torch.randn(2, 5, 16, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None

# ─── Attention ───
def test_attention_basic():
    from bitvideo.models import Attention
    attn = Attention(dim=16, num_heads=4, qk_norm=True).eval()
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape

def test_attention_kv_cache():
    from bitvideo.models import Attention
    attn = Attention(dim=16, num_heads=4, is_causal=True).eval()
    seq = torch.randn(1, 5, 16)
    with torch.no_grad():
        full = attn(seq)
        cache = None
        incremental = []
        for i in range(5):
            out, cache = attn(seq[:, i:i+1], past_key_value=cache, use_cache=True)
            incremental.append(out)
    torch.testing.assert_close(torch.cat(incremental, 1), full, rtol=1e-5, atol=1e-5)

# ─── Attention Variants ───
def test_spatial_attention():
    from bitvideo.models import SpatialAttention
    sa = SpatialAttention(dim=16, num_heads=4).eval()
    x = torch.randn(2, 12, 16)
    out = sa(x, temporal_size=3, spatial_size=4)
    assert out.shape == (2, 12, 16)

def test_temporal_attention():
    from bitvideo.models import TemporalAttention
    ta = TemporalAttention(dim=16, num_heads=4, is_causal=True).eval()
    x = torch.randn(2, 12, 16)
    out = ta(x, temporal_size=3, spatial_size=4)
    assert out.shape == (2, 12, 16)

def test_cross_attention():
    from bitvideo.models import CrossAttention
    ca = CrossAttention(dim=16, num_heads=4, context_dim=12, gate=True).eval()
    x = torch.randn(2, 6, 16)
    ctx = torch.randn(2, 4, 12)
    with torch.no_grad():
        out, cache = ca(x, context=ctx, return_cache=True)
        out2 = ca(x, context_cache=cache)
    assert out.shape == (2, 6, 16)
    assert out2.shape == (2, 6, 16)

# ─── Transformer ───
def test_video_dit():
    from bitvideo.models import VideoDiT
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                     context_dim=24, patch_size=(1,2,2), qk_norm=True).eval()
    video = torch.randn(1, 4, 2, 4, 4)
    t = torch.tensor([100.0])
    ctx = torch.randn(1, 3, 24)
    with torch.no_grad():
        out = model(video, t, ctx)
    assert out.shape == video.shape
    assert torch.isfinite(out).all()

def test_video_dit_gradient():
    from bitvideo.models import VideoDiT
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                     context_dim=24, patch_size=(1,2,2), qk_norm=True)
    video = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
    loss = model(video, torch.tensor([50.0]), torch.randn(1, 3, 24)).square().mean()
    loss.backward()
    assert video.grad is not None and torch.isfinite(video.grad).all()

# ─── Pipeline ───
def test_schedulers():
    from bitvideo.pipeline import (DDIMScheduler, EulerScheduler, EulerAncestralScheduler,
                                    DPMPlusPlusScheduler, PNDMScheduler, UniPCScheduler)
    for Sched in (DDIMScheduler, EulerScheduler, EulerAncestralScheduler,
                  DPMPlusPlusScheduler, PNDMScheduler, UniPCScheduler):
        s = Sched(num_train_steps=100)
        s.set_timesteps(5)
        x = torch.randn(1, 4, 2, 4, 4)
        for t in s.timesteps[:2]:
            x = s.step(torch.randn_like(x), t.item(), x)
        assert x.shape == (1, 4, 2, 4, 4)

def test_decoder():
    from bitvideo.pipeline import VideoVAEDecoder, ChunkedVideoDecoder
    dec = VideoVAEDecoder(in_channels=4, out_channels=3, base_channels=16,
                          channel_multipliers=(2, 1), num_res_blocks=1, num_groups=4)
    out = dec(torch.randn(1, 4, 2, 4, 4))
    assert out.shape[1] == 3
    chunked = ChunkedVideoDecoder(dec, temporal_chunk_size=2, temporal_overlap=1)
    out_long = chunked(torch.randn(1, 4, 5, 4, 4))
    assert out_long.shape[1] == 3

def test_pipeline():
    from bitvideo.models import VideoDiT
    from bitvideo.pipeline import BitVideoPipeline, DDIMScheduler
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                     context_dim=24, patch_size=(1,2,2), qk_norm=True).eval()
    sched = DDIMScheduler()
    pipe = BitVideoPipeline(model, sched, decoder=None)
    out = pipe(torch.randn(1, 3, 24), num_frames=2, height=4, width=4,
              num_inference_steps=2, decode=False)
    assert out.shape == (1, 4, 2, 4, 4)

# ─── Training ───
def test_diffusion_loss():
    from bitvideo.training import DiffusionLoss
    loss_fn = DiffusionLoss(prediction_type="epsilon")
    pred = torch.randn(2, 4, 2, 4, 4)
    target = torch.randn_like(pred)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and torch.isfinite(loss)

def test_synthetic_dataset():
    from bitvideo.training.datasets import SyntheticVideoDataset, create_dataloader
    ds = SyntheticVideoDataset(num_samples=4, latent_channels=4, num_frames=2,
                               height=4, width=4, text_length=3, text_dim=24)
    dl = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert batch["video_latent"].shape == (2, 4, 2, 4, 4)
    assert batch["text_embedding"].shape == (2, 3, 24)

# ─── Extras ───
def test_lora():
    from bitvideo.models import VideoDiT
    from bitvideo.extras import apply_lora
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                     context_dim=24, patch_size=(1,2,2), qk_norm=True)
    layers = apply_lora(model, rank=4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(layers) > 0 and trainable > 0

def test_config():
    import tempfile, os
    from bitvideo.extras import BitVideoConfig, save_config, load_config
    cfg = BitVideoConfig(dim=128, depth=2)
    tmp = tempfile.mktemp(suffix=".json")
    save_config(cfg, tmp)
    loaded = load_config(tmp)
    assert loaded.dim == 128 and loaded.depth == 2
    os.unlink(tmp)

def test_profiling():
    from bitvideo.models import VideoDiT
    from bitvideo.extras import profile_model
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                     context_dim=24, patch_size=(1,2,2), qk_norm=True).eval()
    result = profile_model(model, torch.randn(1,4,2,4,4), torch.tensor([100.0]),
                          torch.randn(1,3,24), backward=False, warmup_steps=1, measure_steps=1)
    assert result.total_params > 0 and result.forward_time_ms > 0

# ─── CUDA (skipped if unavailable) ───
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_video_dit():
    from bitvideo.models import VideoDiT
    device = torch.device("cuda")
    for dtype in (torch.float16, torch.bfloat16):
        model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4,
                         context_dim=24, patch_size=(1,2,2), qk_norm=True,
                         device=device, dtype=dtype).eval()
        with torch.no_grad():
            out = model(torch.randn(1,4,2,4,4, device=device, dtype=dtype),
                       torch.tensor([100.0], device=device),
                       torch.randn(1,3,24, device=device, dtype=dtype))
        assert out.shape == (1,4,2,4,4) and torch.isfinite(out).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pipeline():
    from bitvideo.models import VideoDiT
    from bitvideo.pipeline import BitVideoPipeline, EulerScheduler
    device = torch.device("cuda")
    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4, context_dim=24,
                     patch_size=(1,2,2), qk_norm=True, device=device, dtype=torch.float16).eval()
    pipe = BitVideoPipeline(model, EulerScheduler(), decoder=None)
    out = pipe(torch.randn(1, 3, 24, device=device, dtype=torch.float16),
              num_frames=2, height=4, width=4, num_inference_steps=3, decode=False)
    assert out.shape == (1, 4, 2, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
