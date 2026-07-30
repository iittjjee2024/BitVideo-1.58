"""Unified inference pipeline for BitVideo-1.58 video generation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from bitvideo.models import VideoDiT

from .decoder import ChunkedVideoDecoder, VideoVAEDecoder
from .schedulers import NoiseScheduler


class BitVideoPipeline:
    """End-to-end video generation pipeline combining denoising and decoding.

    Orchestrates the full inference loop:
    1. Generate initial noise in latent space
    2. Iteratively denoise using the Video DiT and a noise scheduler
    3. Decode latents to pixel space using the VAE decoder
    4. Post-process (rescale, clamp)

    This class does NOT subclass nn.Module; it is a stateless orchestrator
    that holds references to the model components.
    """

    def __init__(
        self,
        transformer: VideoDiT,
        scheduler: NoiseScheduler,
        decoder: VideoVAEDecoder | ChunkedVideoDecoder | None = None,
        *,
        latent_scale_factor: float = 0.18215,
    ) -> None:
        if not isinstance(transformer, VideoDiT):
            raise TypeError("transformer must be a VideoDiT instance")
        if not isinstance(scheduler, NoiseScheduler):
            raise TypeError("scheduler must be a NoiseScheduler instance")
        if decoder is not None and not isinstance(decoder, (VideoVAEDecoder, ChunkedVideoDecoder)):
            raise TypeError("decoder must be a VideoVAEDecoder, ChunkedVideoDecoder, or None")
        self.transformer = transformer
        self.scheduler = scheduler
        self.decoder = decoder
        self.latent_scale_factor = float(latent_scale_factor)

    @property
    def device(self) -> torch.device:
        """Return the device of the transformer parameters."""

        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the transformer parameters."""

        return next(self.transformer.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        context: torch.Tensor,
        *,
        num_frames: int = 16,
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        decode: bool = True,
        callback: Any = None,
        callback_steps: int = 1,
    ) -> torch.Tensor:
        """Generate video from text conditioning.

        Args:
            context: Text encoder output ``[B, Lc, context_dim]``.
            num_frames: Number of output video frames.
            height: Spatial height in latent space.
            width: Spatial width in latent space.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance strength (1.0 = no guidance).
            negative_context: Unconditional context for CFG ``[B, Lc, context_dim]``.
            generator: Optional random generator for reproducibility.
            latents: Optional pre-generated initial noise ``[B, C, T, H, W]``.
            decode: Whether to decode latents to pixel space.
            callback: Optional callback(step_idx, timestep, latents) called every callback_steps.
            callback_steps: Frequency of callback invocation.

        Returns:
            Generated video ``[B, C_out, T, H', W']`` in pixel space (if decode=True)
            or latent space ``[B, C_latent, T, H, W]`` (if decode=False).
        """

        if not isinstance(context, torch.Tensor):
            raise TypeError("context must be a torch.Tensor")
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B, Lc, D]; got {tuple(context.shape)}")
        batch_size = context.shape[0]
        device = self.device
        dtype = self.dtype
        context = context.to(device=device, dtype=dtype)

        # Setup scheduler timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Generate or validate initial noise.
        latent_channels = self.transformer.in_channels
        if latents is None:
            # latents: [B, C_latent, T, H, W].
            latents = torch.randn(
                batch_size,
                latent_channels,
                num_frames,
                height,
                width,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            if latents.shape[0] != batch_size:
                raise ValueError("latents batch size must match context")
            latents = latents.to(device=device, dtype=dtype)

        # Prepare for classifier-free guidance.
        use_cfg = guidance_scale > 1.0 and negative_context is not None
        if use_cfg:
            negative_context = negative_context.to(device=device, dtype=dtype)
            # Concatenate negative and positive context: [2B, Lc, D].
            cfg_context = torch.cat([negative_context, context], dim=0)

        # Denoising loop.
        for step_idx, timestep in enumerate(self.scheduler.timesteps):
            t_value = timestep.item()
            # Prepare timestep tensor: [B] or [2B].
            if use_cfg:
                # Duplicate latents for unconditional and conditional.
                latent_input = torch.cat([latents, latents], dim=0)
                t_tensor = torch.full(
                    (2 * batch_size,), t_value, device=device, dtype=dtype
                )
                model_context = cfg_context
            else:
                latent_input = latents
                t_tensor = torch.full((batch_size,), t_value, device=device, dtype=dtype)
                model_context = context

            # Model prediction: [B or 2B, C, T, H, W].
            noise_pred = self.transformer(latent_input, t_tensor, model_context)

            # Apply classifier-free guidance.
            if use_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Scheduler step: denoise.
            latents = self.scheduler.step(
                noise_pred,
                t_value,
                latents,
                generator=generator,
            )

            # Callback.
            if callback is not None and (step_idx + 1) % callback_steps == 0:
                callback(step_idx, t_value, latents)

        # Decode to pixel space if requested.
        if decode and self.decoder is not None:
            # Scale latents.
            scaled_latents = latents / self.latent_scale_factor
            # video: [B, C_out, T', H', W'].
            video = self.decoder(scaled_latents)
            # Rescale to [0, 1].
            video = (video + 1.0) / 2.0
            video = video.clamp(0.0, 1.0)
            return video

        return latents

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BitVideoPipeline":
        """Move all components to a device/dtype."""

        if device is not None or dtype is not None:
            kwargs: dict[str, Any] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            self.transformer = self.transformer.to(**kwargs)
            if self.decoder is not None:
                self.decoder = self.decoder.to(**kwargs)
        return self

    def eval(self) -> "BitVideoPipeline":
        """Set all components to evaluation mode."""

        self.transformer.eval()
        if self.decoder is not None:
            self.decoder.eval()
        return self

    @torch.no_grad()
    def pack_weights(self, **kwargs) -> None:
        """Pack transformer weights for quantized inference."""

        self.transformer.pack_weights(**kwargs)
