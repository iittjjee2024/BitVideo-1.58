"""BitMem training harness.

Stage 1: FP16 baseline trainer. Stage 2: ternary via config toggle.
Later stages add memory adapters, agent controller, and joint optimization.
"""

from bitmem.train.synthetic import (
    SyntheticDiffusionDataset,
    make_synthetic_batch,
)
from bitmem.train.baseline import (
    BaselineConfig,
    BaselineTrainer,
    QuantMode,
    quantization_for_mode,
)

__all__ = [
    "BaselineConfig",
    "BaselineTrainer",
    "QuantMode",
    "SyntheticDiffusionDataset",
    "make_synthetic_batch",
    "quantization_for_mode",
]
