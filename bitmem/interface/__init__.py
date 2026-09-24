"""Memory-to-DiT injection interfaces (§5): Method A/B/C.

Stage 0 ships Method B (memory tokens) as the simplest to validate. Methods A
(cross-attention) and C (adaptive conditioning) arrive at Stage 3 and are
compared in the ablation matrix (§13).
"""

from bitmem.interface.mem_tokens import MemoryTokenInterface

__all__ = ["MemoryTokenInterface"]
