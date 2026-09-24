"""BitMem evaluation suite (§12).

Generation, memory, agent, and efficiency metrics used across all stages and
by the ablation matrix (§13).
"""

from bitmem.eval.metrics import (
    EfficiencyReport,
    count_parameters,
    denoising_mse,
    measure_efficiency,
    ternary_degradation,
)

__all__ = [
    "EfficiencyReport",
    "count_parameters",
    "denoising_mse",
    "measure_efficiency",
    "ternary_degradation",
]
