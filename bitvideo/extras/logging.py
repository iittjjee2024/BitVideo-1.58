"""Structured logging utilities for BitVideo training and inference."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any


def setup_logging(
    *,
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure the root logger with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
        format_string: Custom format string.

    Returns:
        Configured root logger.
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("bitvideo")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates.
    logger.handlers.clear()

    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # Optional file handler.
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Structured training metrics logger with step tracking.

    Accumulates metrics over steps and provides periodic summaries.
    Designed for integration with training loops.
    """

    def __init__(
        self,
        *,
        log_every_steps: int = 100,
        logger: logging.Logger | None = None,
    ) -> None:
        self.log_every_steps = max(1, log_every_steps)
        self.logger = logger or logging.getLogger("bitvideo.training")
        self.step = 0
        self.epoch = 0
        self._metrics: dict[str, list[float]] = {}
        self._start_time = time.time()
        self._step_start_time = time.time()

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Record metrics for a training step.

        Args:
            step: Current global step.
            metrics: Dict of metric name -> value.
        """
        self.step = step
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

        if step % self.log_every_steps == 0 and step > 0:
            self._flush()

    def log_epoch(self, epoch: int) -> None:
        """Mark an epoch boundary."""
        self.epoch = epoch
        self.logger.info(f"Epoch {epoch} complete")

    def _flush(self) -> None:
        """Write accumulated metrics to the logger."""
        elapsed = time.time() - self._start_time
        parts = [f"step={self.step}"]
        for key, values in self._metrics.items():
            if values:
                avg = sum(values[-self.log_every_steps:]) / min(
                    len(values), self.log_every_steps
                )
                parts.append(f"{key}={avg:.4f}")
        parts.append(f"elapsed={elapsed:.1f}s")
        self.logger.info(" | ".join(parts))

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "metrics": {k: v[-100:] for k, v in self._metrics.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self.step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)
        self._metrics = state.get("metrics", {})
