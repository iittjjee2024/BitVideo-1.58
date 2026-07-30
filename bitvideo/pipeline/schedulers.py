"""Diffusion noise schedulers for BitVideo-1.58 inference and training."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import torch


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cosine_beta_schedule(num_train_steps: int, *, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from 'Improved Denoising Diffusion Probabilistic Models'."""

    # steps: [num_train_steps+1] float64 for precision.
    steps = torch.arange(num_train_steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / num_train_steps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # betas: [num_train_steps] clipped.
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999).float()


def _linear_beta_schedule(
    num_train_steps: int,
    *,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> torch.Tensor:
    """Linear noise schedule scaled for latent diffusion."""

    # betas: [num_train_steps].
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_steps).square()


class NoiseScheduler(ABC):
    """Base class for all diffusion noise schedulers.

    Provides the common interface for adding noise (forward process) and
    removing noise (reverse/sampling process).
    """

    def __init__(
        self,
        num_train_steps: int = 1000,
        *,
        beta_schedule: str = "linear",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
    ) -> None:
        self.num_train_steps = _positive_int(num_train_steps, "num_train_steps")
        if prediction_type not in {"epsilon", "v_prediction", "sample"}:
            raise ValueError(
                f"prediction_type must be epsilon, v_prediction, or sample; "
                f"got {prediction_type!r}"
            )
        self.prediction_type = prediction_type
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)

        # Compute noise schedule.
        schedule = beta_schedule.strip().lower()
        if schedule == "linear":
            betas = _linear_beta_schedule(
                num_train_steps, beta_start=beta_start, beta_end=beta_end
            )
        elif schedule == "cosine":
            betas = _cosine_beta_schedule(num_train_steps)
        else:
            raise ValueError(f"beta_schedule must be linear or cosine; got {beta_schedule!r}")

        # Precompute cumulative products: all [num_train_steps].
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1.0).sqrt()

        # Inference timesteps (set by set_timesteps).
        self.timesteps: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.num_inference_steps: int = 0

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        """Set the discrete timesteps for sampling."""

        self.num_inference_steps = _positive_int(num_inference_steps, "num_inference_steps")
        # Uniform spacing from T-1 down to 0.
        step_ratio = self.num_train_steps / num_inference_steps
        # timesteps: [num_inference_steps] long, descending.
        self.timesteps = (
            torch.arange(num_inference_steps, dtype=torch.float64) * step_ratio
        ).round().flip(0).long().to(device)

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion process: add noise at given timesteps.

        Args:
            original: Clean sample ``[B, ...]``.
            noise: Gaussian noise ``[B, ...]`` same shape as original.
            timesteps: Integer timesteps ``[B]``.

        Returns:
            Noisy sample ``[B, ...]``.
        """

        # sqrt_alpha: [B, 1, 1, ...]; sqrt_one_minus_alpha: [B, 1, 1, ...].
        sqrt_alpha = self._gather_and_expand(self.sqrt_alphas_cumprod, timesteps, original)
        sqrt_one_minus_alpha = self._gather_and_expand(
            self.sqrt_one_minus_alphas_cumprod, timesteps, original
        )
        # noisy: original.shape.
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target for v-prediction training.

        Args:
            sample: Clean sample ``[B, ...]``.
            noise: Noise ``[B, ...]``.
            timesteps: Integer timesteps ``[B]``.

        Returns:
            Velocity ``[B, ...]``.
        """

        sqrt_alpha = self._gather_and_expand(self.sqrt_alphas_cumprod, timesteps, sample)
        sqrt_one_minus_alpha = self._gather_and_expand(
            self.sqrt_one_minus_alphas_cumprod, timesteps, sample
        )
        # velocity: sample.shape.
        return sqrt_alpha * noise - sqrt_one_minus_alpha * sample

    def _predict_original(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Convert model output to predicted original sample."""

        alpha_prod = self.alphas_cumprod[timestep]
        sqrt_alpha = alpha_prod.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha_prod).sqrt()

        if self.prediction_type == "epsilon":
            # x0 = (xt - sqrt(1-alpha) * eps) / sqrt(alpha).
            predicted = (sample - sqrt_one_minus_alpha * model_output) / sqrt_alpha
        elif self.prediction_type == "v_prediction":
            # x0 = sqrt(alpha) * xt - sqrt(1-alpha) * v.
            predicted = sqrt_alpha * sample - sqrt_one_minus_alpha * model_output
        elif self.prediction_type == "sample":
            predicted = model_output
        else:
            raise ValueError(f"unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            predicted = predicted.clamp(-self.clip_sample_range, self.clip_sample_range)
        return predicted

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Single denoising step. Returns the previous sample."""

        ...

    @staticmethod
    def _gather_and_expand(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: torch.Tensor,
    ) -> torch.Tensor:
        """Gather schedule values at timesteps and reshape for broadcasting."""

        # gathered: [B].
        gathered = values.to(timesteps.device).gather(0, timesteps.long())
        # Expand to match sample dimensions: [B, 1, 1, ...].
        while gathered.ndim < broadcast_shape.ndim:
            gathered = gathered.unsqueeze(-1)
        return gathered.to(broadcast_shape.dtype)


class DDIMScheduler(NoiseScheduler):
    """Denoising Diffusion Implicit Models (DDIM) scheduler.

    Deterministic sampling with optional eta parameter for stochasticity.
    """

    def __init__(
        self,
        num_train_steps: int = 1000,
        *,
        eta: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(num_train_steps, **kwargs)
        if not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be a non-negative finite number")
        self.eta = float(eta)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDIM denoising step."""

        # Current and previous alphas.
        alpha_prod_t = self.alphas_cumprod[timestep]
        # Find the previous timestep index.
        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if step_idx.numel() == 0:
            alpha_prod_prev = torch.tensor(1.0)
        elif step_idx.item() + 1 < len(self.timesteps):
            prev_t = self.timesteps[step_idx.item() + 1].item()
            alpha_prod_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_prev = torch.tensor(1.0)

        # Predict x0.
        predicted_original = self._predict_original(model_output, sample, timestep)

        # Compute "direction pointing to xt".
        # sigma: scalar.
        sigma = self.eta * (
            (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t) * (1.0 - alpha_prod_t / alpha_prod_prev)
        ).sqrt()
        # pred_direction: sample.shape.
        pred_direction = (1.0 - alpha_prod_prev - sigma**2).sqrt() * (
            (sample - alpha_prod_t.sqrt() * predicted_original) / (1.0 - alpha_prod_t).sqrt()
        )
        # prev_sample: sample.shape.
        prev_sample = alpha_prod_prev.sqrt() * predicted_original + pred_direction
        if self.eta > 0.0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + sigma * noise
        return prev_sample


class EulerScheduler(NoiseScheduler):
    """Euler method (first-order ODE solver) for diffusion sampling."""

    def __init__(self, num_train_steps: int = 1000, **kwargs) -> None:
        super().__init__(num_train_steps, **kwargs)
        # Precompute sigmas for the continuous-time formulation.
        self.sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        super().set_timesteps(num_inference_steps, device=device)
        # sigmas_schedule: [num_inference_steps+1], with trailing 0.
        timestep_indices = self.timesteps.cpu()
        self.sigmas_schedule = torch.cat([
            self.sigmas[timestep_indices],
            torch.zeros(1),
        ]).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler denoising step."""

        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        sigma = self.sigmas_schedule[step_idx]
        sigma_next = self.sigmas_schedule[step_idx + 1]

        # Convert to the noise-scaled space.
        predicted_original = self._predict_original(model_output, sample, timestep)

        # d(sample)/d(sigma) direction.
        # derivative: sample.shape.
        derivative = (sample - predicted_original) / sigma
        # dt: scalar (sigma_next - sigma).
        dt = sigma_next - sigma
        # prev_sample: sample.shape.
        return sample + derivative * dt


class EulerAncestralScheduler(NoiseScheduler):
    """Euler Ancestral (stochastic) scheduler with noise injection."""

    def __init__(self, num_train_steps: int = 1000, *, eta: float = 1.0, **kwargs) -> None:
        super().__init__(num_train_steps, **kwargs)
        self.eta = float(eta)
        self.sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        super().set_timesteps(num_inference_steps, device=device)
        timestep_indices = self.timesteps.cpu()
        self.sigmas_schedule = torch.cat([
            self.sigmas[timestep_indices],
            torch.zeros(1),
        ]).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler Ancestral denoising step with noise injection."""

        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        sigma = self.sigmas_schedule[step_idx]
        sigma_next = self.sigmas_schedule[step_idx + 1]

        predicted_original = self._predict_original(model_output, sample, timestep)

        # Compute sigma_down and sigma_up for ancestral sampling.
        sigma_up = (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2).sqrt() * self.eta
        sigma_down = (sigma_next**2 - sigma_up**2).sqrt()

        # Euler step to sigma_down.
        derivative = (sample - predicted_original) / sigma
        sample_down = sample + derivative * (sigma_down - sigma)

        # Add noise scaled by sigma_up.
        if sigma_up > 0.0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            sample_down = sample_down + sigma_up * noise
        return sample_down


class DPMPlusPlusScheduler(NoiseScheduler):
    """DPM-Solver++ (2nd order) scheduler for fast high-quality sampling."""

    def __init__(self, num_train_steps: int = 1000, *, solver_order: int = 2, **kwargs) -> None:
        super().__init__(num_train_steps, **kwargs)
        if solver_order not in {1, 2, 3}:
            raise ValueError("solver_order must be 1, 2, or 3")
        self.solver_order = solver_order
        self._model_outputs: list[torch.Tensor] = []

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        super().set_timesteps(num_inference_steps, device=device)
        self._model_outputs = []

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DPM-Solver++ denoising step."""

        predicted_original = self._predict_original(model_output, sample, timestep)
        self._model_outputs.append(predicted_original)
        if len(self._model_outputs) > self.solver_order:
            self._model_outputs.pop(0)

        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        if step_idx + 1 < len(self.timesteps):
            prev_t = self.timesteps[step_idx + 1].item()
        else:
            prev_t = 0

        # Log-SNR values.
        lambda_t = torch.log(self.alphas_cumprod[timestep].sqrt() / (1.0 - self.alphas_cumprod[timestep]).sqrt())
        lambda_prev = torch.log(self.alphas_cumprod[prev_t].sqrt() / (1.0 - self.alphas_cumprod[prev_t]).sqrt())
        h = lambda_prev - lambda_t

        alpha_prev = self.alphas_cumprod[prev_t].sqrt()
        sigma_prev = (1.0 - self.alphas_cumprod[prev_t]).sqrt()

        if len(self._model_outputs) == 1 or self.solver_order == 1:
            # First-order update.
            prev_sample = alpha_prev * predicted_original + sigma_prev * (
                (sample - self.alphas_cumprod[timestep].sqrt() * predicted_original)
                / (1.0 - self.alphas_cumprod[timestep]).sqrt()
            ) * torch.exp(-h)
        else:
            # Second-order update using previous prediction.
            x0_prev = self._model_outputs[-2]
            # Linear interpolation correction.
            correction = 0.5 * (predicted_original - x0_prev)
            prev_sample = (
                alpha_prev * (predicted_original + correction * (1.0 - torch.exp(-h)))
                + sigma_prev * (
                    (sample - self.alphas_cumprod[timestep].sqrt() * predicted_original)
                    / (1.0 - self.alphas_cumprod[timestep]).sqrt()
                ) * torch.exp(-h)
            )
        return prev_sample


class PNDMScheduler(NoiseScheduler):
    """Pseudo Numerical methods for Diffusion Models (PNDM/PLMS) scheduler."""

    def __init__(self, num_train_steps: int = 1000, **kwargs) -> None:
        super().__init__(num_train_steps, **kwargs)
        self._ets: list[torch.Tensor] = []

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        super().set_timesteps(num_inference_steps, device=device)
        self._ets = []

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """PNDM denoising step using linear multi-step method."""

        self._ets.append(model_output)
        if len(self._ets) > 4:
            self._ets.pop(0)

        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        if step_idx + 1 < len(self.timesteps):
            prev_t = self.timesteps[step_idx + 1].item()
        else:
            prev_t = 0

        # Use linear multi-step coefficients.
        if len(self._ets) == 1:
            eps_prime = self._ets[-1]
        elif len(self._ets) == 2:
            eps_prime = (3.0 * self._ets[-1] - self._ets[-2]) / 2.0
        elif len(self._ets) == 3:
            eps_prime = (23.0 * self._ets[-1] - 16.0 * self._ets[-2] + 5.0 * self._ets[-3]) / 12.0
        else:
            eps_prime = (
                55.0 * self._ets[-1] - 59.0 * self._ets[-2] + 37.0 * self._ets[-3] - 9.0 * self._ets[-4]
            ) / 24.0

        # DDIM-like step with the corrected eps.
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev = self.alphas_cumprod[prev_t]
        predicted_original = self._predict_original(eps_prime, sample, timestep)
        pred_direction = (1.0 - alpha_prod_prev).sqrt() * eps_prime
        return alpha_prod_prev.sqrt() * predicted_original + pred_direction


class UniPCScheduler(NoiseScheduler):
    """Unified Predictor-Corrector (UniPC) scheduler for fast sampling."""

    def __init__(self, num_train_steps: int = 1000, *, solver_order: int = 2, **kwargs) -> None:
        super().__init__(num_train_steps, **kwargs)
        if solver_order not in {1, 2, 3}:
            raise ValueError("solver_order must be 1, 2, or 3")
        self.solver_order = solver_order
        self._model_outputs: list[torch.Tensor] = []

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device | str = "cpu") -> None:
        super().set_timesteps(num_inference_steps, device=device)
        self._model_outputs = []

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """UniPC denoising step (predictor phase)."""

        predicted_original = self._predict_original(model_output, sample, timestep)
        self._model_outputs.append(predicted_original)
        if len(self._model_outputs) > self.solver_order:
            self._model_outputs.pop(0)

        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        if step_idx + 1 < len(self.timesteps):
            prev_t = self.timesteps[step_idx + 1].item()
        else:
            prev_t = 0

        alpha_prev = self.alphas_cumprod[prev_t].sqrt()
        sigma_prev = (1.0 - self.alphas_cumprod[prev_t]).sqrt()
        alpha_t = self.alphas_cumprod[timestep].sqrt()
        sigma_t = (1.0 - self.alphas_cumprod[timestep]).sqrt()

        # First-order predictor (Euler in log-SNR space).
        lambda_t = torch.log(alpha_t / sigma_t)
        lambda_prev = torch.log(alpha_prev / sigma_prev)
        h = lambda_prev - lambda_t

        # x_{t-1} via exponential integrator.
        prev_sample = (alpha_prev / alpha_t) * sample - sigma_prev * (torch.exp(h) - 1.0) * (
            (sample - alpha_t * predicted_original) / sigma_t
        )

        # Higher-order correction using previous predictions.
        if len(self._model_outputs) >= 2 and self.solver_order >= 2:
            correction = predicted_original - self._model_outputs[-2]
            prev_sample = prev_sample - 0.5 * sigma_prev * (torch.exp(h) - 1.0) * correction / sigma_t

        return prev_sample
