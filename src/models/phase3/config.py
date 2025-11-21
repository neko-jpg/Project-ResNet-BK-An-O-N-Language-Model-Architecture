"""
Configuration for Phase 3: Physics Transcendence

This module defines the configuration class for Phase 3 models.
"""

from typing import Optional
from dataclasses import dataclass

@dataclass
class Phase3Config:
    """
    Configuration class for Phase 3 Integrated Model.

    Args:
        vocab_size (int): Vocabulary size.
        d_model (int): Model dimension.
        n_layers (int): Number of Phase 3 blocks.
        max_seq_len (int): Maximum sequence length.
        use_complex32 (bool): Whether to use complex32 (float16 complex) for memory efficiency.
        d_koopman (int): Dimension of Koopman space.
        potential_type (str): Type of potential function ('bk_core' or 'mlp').
        dropout (float): Dropout rate.
        zeta_scale (float): Scale for Zeta initialization.
        trainable_pos (bool): Whether position embeddings are trainable.
    """
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int = 2048
    use_complex32: bool = True
    d_koopman: Optional[int] = None
    potential_type: str = 'bk_core'
    dropout: float = 0.1
    zeta_scale: float = 1.0
    trainable_pos: bool = False

    def __post_init__(self):
        if self.d_koopman is None:
            self.d_koopman = self.d_model * 2
