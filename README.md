# BitVideo-1.58

**Production-grade W1.58A8 Video Diffusion Transformer** — a complete implementation of ternary quantized video generation using 1.58-bit weights and 8-bit activations.

## Overview

BitVideo-1.58 implements the BitNet b1.58 quantization scheme (ternary weights {-1, 0, +1} with per-tensor absmean scaling) applied to a Video Diffusion Transformer architecture. Every linear projection uses quantization-aware training (QAT) with straight-through estimators, enabling massive memory and compute savings while maintaining generation quality.

## Key Features

- **W1.58A8 Quantization**: All linear layers use ternary weights (1.58 bits) with INT8 activations
- **Custom CUDA Kernels**: Optimized GEMV, DP4A GEMM, and MMA tensor-core kernels for ternary computation
- **3-Tier Dispatch**: Automatic backend selection (CUDA extension > Triton > PyTorch fallback)
- **Video DiT Architecture**: Spatial + temporal factorized attention with adaLN-Zero conditioning
- **6 Noise Schedulers**: DDIM, Euler, Euler Ancestral, DPM-Solver++, PNDM, UniPC
- **Training Stack**: QAT trainer with min-SNR weighting, knowledge distillation, LoRA fine-tuning
- **Production Ready**: ControlNet, ONNX/TorchScript export, profiling, structured logging

## Architecture

```
Video [B,C,T,H,W]
    │
    ▼ Conv3D Patch Embedding
Tokens [B, T*H*W, D]
    │
    ▼ N × VideoDiTBlock
    │   ├── adaLN-Zero → Spatial Self-Attention → gate → residual
    │   ├── adaLN-Zero → Temporal Self-Attention → gate → residual
    │   ├── adaLN → Cross-Attention (text) → residual
    │   └── adaLN-Zero → SwiGLU FFN → gate → residual
    │
    ▼ Final Layer (adaLN + Linear)
Prediction [B, T*H*W, C*patch_vol]
    │
    ▼ Unpatchify
Output [B,C,T,H,W]
```

## Installation

```bash
# Basic installation (PyTorch fallback)
pip install -e .

# With CUDA extension (requires nvcc + CUDA 12+)
pip install -e .
```

## Quick Start

```python
import torch
from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DDIMScheduler

# Create model
model = VideoDiT(
    in_channels=4, dim=768, depth=12, num_heads=12,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
)

# Setup pipeline
scheduler = DDIMScheduler(num_train_steps=1000)
pipeline = BitVideoPipeline(model, scheduler)

# Generate (with pre-computed text embeddings)
context = torch.randn(1, 77, 768)  # From text encoder
video = pipeline(context, num_frames=16, height=32, width=32,
                 num_inference_steps=50)
```

## Training

```python
from bitvideo.training.train_qat_video import TrainingConfig, train_qat

config = TrainingConfig(
    dim=768, depth=12, num_heads=12,
    learning_rate=1e-4, max_steps=100_000,
    mixed_precision="bf16",
)
train_qat(config)
```

## Project Structure

```
bitvideo/
├── cuda/           # CUDA extension (GEMV, DP4A, MMA kernels)
├── ops/            # Python operator layer (3-tier dispatch)
├── quantization/   # STE, quantizers, BitLinear
├── models/         # RoPE, patch embed, attention, Video DiT
├── pipeline/       # Schedulers, VAE decoder, inference pipeline
├── training/       # Losses, distillation, datasets, QAT trainer
├── triton/         # Optional Triton kernel backends
└── extras/         # LoRA, ControlNet, export, profiling, config
tests/              # 22 pytest integration tests
benchmarks/         # Inference throughput benchmarks
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (2.8+ recommended for full torch.compile support)
- CUDA 12+ (optional, for native kernels)

## Benchmarks (RTX 5050 Laptop GPU, FP16)

| Config | Parameters | Forward (ms) | Throughput |
|--------|-----------|-------------|-----------|
| Tiny (D=64, L=2) | 272K | 75ms | 13.3/s |
| Small (D=128, L=4) | 2.0M | 145ms | 6.9/s |
| Base (D=256, L=6) | 11.7M | 198ms | 5.1/s |

## References

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)
- BitNet: Scaling 1-bit Transformers for Large Language Models (2023)

## License

Research use. See LICENSE for details.
