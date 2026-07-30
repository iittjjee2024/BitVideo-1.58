"""Loss functions for diffusion model training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class DiffusionLoss(nn.Module):
    """Standard diffusion training loss with configurable prediction target.

    Supports epsilon prediction, v-prediction, and direct sample prediction
    with optional per-timestep SNR weighting.
    """

    def __init__(
        self,
        *,
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
        snr_gamma: float | None = None,
    ) -> None:
        super().__init__()
        if prediction_type not in {"epsilon", "v_prediction", "sample"}:
            raise ValueError(
                f"prediction_type must be epsilon, v_prediction, or sample; "
                f"got {prediction_type!r}"
            )
        if loss_type not in {"mse", "l1", "huber"}:
            raise ValueError(f"loss_type must be mse, l1, or huber; got {loss_type!r}")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        alphas_cumprod: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            model_output: Model prediction ``[B, ...]``.
            target: Ground truth target ``[B, ...]`` (noise, velocity, or sample).
            timesteps: Integer timesteps ``[B]`` for SNR weighting.
            alphas_cumprod: Full schedule ``[T]`` for SNR computation.

        Returns:
            Scalar loss.
        """

        if model_output.shape != target.shape:
            raise ValueError(
                f"model_output and target shapes must match; got "
                f"{tuple(model_output.shape)} and {tuple(target.shape)}"
            )

        # Per-element loss: [B, ...].
        if self.loss_type == "mse":
            per_element = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "l1":
            per_element = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            per_element = F.huber_loss(model_output, target, reduction="none", delta=1.0)
        else:
            raise ValueError(f"unknown loss_type: {self.loss_type}")

        # Per-sample loss: [B].
        # Flatten spatial/channel dims and average.
        batch_size = per_element.shape[0]
        per_sample = per_element.reshape(batch_size, -1).mean(dim=1)

        # Optional min-SNR-gamma weighting (from "Efficient Diffusion Training").
        if self.snr_gamma is not None and timesteps is not None and alphas_cumprod is not None:
            # snr: [B].
            snr = alphas_cumprod[timesteps] / (1.0 - alphas_cumprod[timesteps])
            # weight: [B], min(snr, gamma) / snr for epsilon, min(snr, gamma) / (snr+1) for v.
            gamma = self.snr_gamma
            if self.prediction_type == "epsilon":
                weight = torch.clamp(snr, max=gamma) / snr
            elif self.prediction_type == "v_prediction":
                weight = torch.clamp(snr, max=gamma) / (snr + 1.0)
            else:
                weight = torch.ones_like(snr)
            per_sample = per_sample * weight.to(per_sample.dtype)

        return per_sample.mean()

    def extra_repr(self) -> str:
        return (
            f"prediction_type={self.prediction_type!r}, loss_type={self.loss_type!r}, "
            f"snr_gamma={self.snr_gamma}"
        )


class SNRWeightedLoss(DiffusionLoss):
    """Convenience alias for DiffusionLoss with min-SNR-gamma weighting enabled."""

    def __init__(
        self,
        *,
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
        snr_gamma: float = 5.0,
    ) -> None:
        super().__init__(
            prediction_type=prediction_type,
            loss_type=loss_type,
            snr_gamma=snr_gamma,
        )


class PerceptualLoss(nn.Module):
    """Simple perceptual loss using feature matching from a frozen encoder.

    Computes L1 distance between intermediate features of a pretrained
    network. This is a lightweight implementation that accepts pre-extracted
    features to avoid bundling a large pretrained model.
    """

    def __init__(self, *, weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be non-negative and finite")
        self.weight = float(weight)

    def forward(
        self,
        predicted_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute perceptual feature matching loss.

        Args:
            predicted_features: List of feature tensors from the prediction.
            target_features: List of feature tensors from the target.

        Returns:
            Weighted scalar loss.
        """

        if len(predicted_features) != len(target_features):
            raise ValueError("predicted and target feature lists must have equal length")
        if not predicted_features:
            raise ValueError("feature lists must not be empty")

        total_loss = torch.tensor(0.0, device=predicted_features[0].device, dtype=predicted_features[0].dtype)
        for pred_feat, tgt_feat in zip(predicted_features, target_features):
            if pred_feat.shape != tgt_feat.shape:
                raise ValueError("predicted and target feature shapes must match")
            total_loss = total_loss + F.l1_loss(pred_feat, tgt_feat)
        return self.weight * total_loss / len(predicted_features)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss wrapper.

    Accepts pre-computed perceptual features (from VGG/AlexNet) and computes
    the weighted channel-wise distance. For full LPIPS, extract features
    externally and pass them here.
    """

    def __init__(self, *, weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be non-negative and finite")
        self.weight = float(weight)

    def forward(
        self,
        predicted_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
        *,
        channel_weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute LPIPS-style perceptual distance.

        Args:
            predicted_features: Normalized feature maps from prediction.
            target_features: Normalized feature maps from target.
            channel_weights: Optional learned per-channel weights for each layer.

        Returns:
            Weighted scalar loss.
        """

        if len(predicted_features) != len(target_features):
            raise ValueError("predicted and target feature lists must have equal length")
        if not predicted_features:
            raise ValueError("feature lists must not be empty")

        total_loss = torch.tensor(0.0, device=predicted_features[0].device, dtype=predicted_features[0].dtype)
        for i, (pred_feat, tgt_feat) in enumerate(zip(predicted_features, target_features)):
            # diff: per-channel squared difference, spatially averaged.
            diff = (pred_feat - tgt_feat).square()
            if channel_weights is not None and i < len(channel_weights):
                # weight: [1, C, 1, ...] broadcast.
                w = channel_weights[i]
                while w.ndim < diff.ndim:
                    w = w.unsqueeze(-1)
                diff = diff * w
            # Spatial mean then channel sum: scalar per batch element.
            spatial_dims = tuple(range(2, diff.ndim))
            total_loss = total_loss + diff.mean(dim=spatial_dims).sum(dim=1).mean()

        return self.weight * total_loss / len(predicted_features)
