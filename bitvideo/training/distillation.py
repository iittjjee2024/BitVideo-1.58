"""Knowledge distillation losses for compressing full-precision teachers into BitVideo."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitDistillationLoss(nn.Module):
    """KL-divergence based logit distillation from a teacher model.

    For diffusion models, this operates on the noise/velocity predictions
    rather than classification logits. The teacher's soft targets guide the
    quantized student.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        weight: float = 1.0,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be non-negative and finite")
        if loss_type not in {"mse", "l1", "kl", "huber"}:
            raise ValueError(f"loss_type must be mse, l1, kl, or huber; got {loss_type!r}")
        self.temperature = float(temperature)
        self.weight = float(weight)
        self.loss_type = loss_type

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss between student and teacher predictions.

        Args:
            student_output: Student model prediction ``[B, ...]``.
            teacher_output: Teacher model prediction ``[B, ...]`` (detached).

        Returns:
            Weighted scalar distillation loss.
        """

        if student_output.shape != teacher_output.shape:
            raise ValueError(
                f"student and teacher output shapes must match; got "
                f"{tuple(student_output.shape)} and {tuple(teacher_output.shape)}"
            )
        teacher_output = teacher_output.detach()

        if self.loss_type == "mse":
            loss = F.mse_loss(student_output / self.temperature, teacher_output / self.temperature)
        elif self.loss_type == "l1":
            loss = F.l1_loss(student_output / self.temperature, teacher_output / self.temperature)
        elif self.loss_type == "huber":
            loss = F.huber_loss(
                student_output / self.temperature,
                teacher_output / self.temperature,
                delta=1.0,
            )
        elif self.loss_type == "kl":
            # Flatten to [B, -1] and apply softmax for KL divergence.
            batch_size = student_output.shape[0]
            student_flat = (student_output / self.temperature).reshape(batch_size, -1)
            teacher_flat = (teacher_output / self.temperature).reshape(batch_size, -1)
            student_log_prob = F.log_softmax(student_flat, dim=-1)
            teacher_prob = F.softmax(teacher_flat, dim=-1)
            loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (
                self.temperature**2
            )
        else:
            raise ValueError(f"unknown loss_type: {self.loss_type}")

        return self.weight * loss

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature:g}, weight={self.weight:g}, "
            f"loss_type={self.loss_type!r}"
        )


class FeatureDistillationLoss(nn.Module):
    """Intermediate feature matching distillation loss.

    Aligns intermediate representations between teacher and student models
    to transfer structural knowledge beyond final predictions.
    """

    def __init__(
        self,
        *,
        weight: float = 1.0,
        loss_type: str = "mse",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be non-negative and finite")
        if loss_type not in {"mse", "l1", "cosine"}:
            raise ValueError(f"loss_type must be mse, l1, or cosine; got {loss_type!r}")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be bool")
        self.weight = float(weight)
        self.loss_type = loss_type
        self.normalize = normalize

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature distillation loss across multiple layers.

        Args:
            student_features: List of student intermediate features.
            teacher_features: List of teacher intermediate features (detached).

        Returns:
            Weighted scalar loss.
        """

        if len(student_features) != len(teacher_features):
            raise ValueError("student and teacher feature lists must have equal length")
        if not student_features:
            raise ValueError("feature lists must not be empty")

        total_loss = torch.tensor(
            0.0,
            device=student_features[0].device,
            dtype=student_features[0].dtype,
        )

        for student_feat, teacher_feat in zip(student_features, teacher_features):
            teacher_feat = teacher_feat.detach()
            if student_feat.shape != teacher_feat.shape:
                raise ValueError(
                    f"student and teacher feature shapes must match; got "
                    f"{tuple(student_feat.shape)} and {tuple(teacher_feat.shape)}"
                )

            if self.normalize:
                # L2 normalize along the channel/feature dimension.
                student_feat = F.normalize(student_feat, dim=1, p=2)
                teacher_feat = F.normalize(teacher_feat, dim=1, p=2)

            if self.loss_type == "mse":
                layer_loss = F.mse_loss(student_feat, teacher_feat)
            elif self.loss_type == "l1":
                layer_loss = F.l1_loss(student_feat, teacher_feat)
            elif self.loss_type == "cosine":
                # Cosine similarity loss: 1 - cos_sim.
                cos_sim = F.cosine_similarity(
                    student_feat.flatten(1),
                    teacher_feat.flatten(1),
                    dim=1,
                )
                layer_loss = (1.0 - cos_sim).mean()
            else:
                raise ValueError(f"unknown loss_type: {self.loss_type}")

            total_loss = total_loss + layer_loss

        return self.weight * total_loss / len(student_features)

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, loss_type={self.loss_type!r}, "
            f"normalize={self.normalize}"
        )


class DistillationLoss(nn.Module):
    """Combined distillation loss: task loss + logit distillation + feature matching.

    Computes a weighted combination of:
    1. Standard diffusion loss (student vs ground truth)
    2. Output-level distillation (student vs teacher prediction)
    3. Feature-level distillation (student vs teacher intermediates)
    """

    def __init__(
        self,
        *,
        task_weight: float = 1.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        temperature: float = 1.0,
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        from .losses import DiffusionLoss

        self.task_loss = DiffusionLoss(prediction_type=prediction_type, loss_type=loss_type)
        self.logit_loss = LogitDistillationLoss(
            temperature=temperature, weight=logit_weight, loss_type=loss_type
        )
        self.feature_loss = FeatureDistillationLoss(weight=feature_weight)
        self.task_weight = float(task_weight)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
        *,
        student_features: list[torch.Tensor] | None = None,
        teacher_features: list[torch.Tensor] | None = None,
        timesteps: torch.Tensor | None = None,
        alphas_cumprod: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined distillation loss.

        Args:
            student_output: Student prediction ``[B, ...]``.
            teacher_output: Teacher prediction ``[B, ...]``.
            target: Ground truth (noise/velocity) ``[B, ...]``.
            student_features: Optional student intermediate features.
            teacher_features: Optional teacher intermediate features.
            timesteps: Integer timesteps for SNR weighting.
            alphas_cumprod: Schedule for SNR computation.

        Returns:
            Dict with 'total', 'task', 'logit', and 'feature' loss tensors.
        """

        # Task loss: student vs ground truth.
        task = self.task_weight * self.task_loss(
            student_output, target, timesteps=timesteps, alphas_cumprod=alphas_cumprod
        )
        # Logit distillation: student vs teacher.
        logit = self.logit_loss(student_output, teacher_output)
        # Feature distillation (if features provided).
        if student_features is not None and teacher_features is not None:
            feature = self.feature_loss(student_features, teacher_features)
        else:
            feature = torch.tensor(0.0, device=student_output.device, dtype=student_output.dtype)
        total = task + logit + feature
        return {"total": total, "task": task, "logit": logit, "feature": feature}
