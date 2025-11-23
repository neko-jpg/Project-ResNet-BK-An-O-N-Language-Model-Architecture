"""
Koopman Operator Model Configuration
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class KoopmanConfig:
    """
    Configuration for KoopmanBKModel.

    Defaults to the "Infinite Width, Single Layer" configuration.
    """
    vocab_size: int = 50257
    d_model: int = 100000 # Infinite width!
    n_layers: int = 1     # Single layer!
    n_seq: int = 128
    num_experts: int = 4
    top_k: int = 1
    dropout_p: float = 0.1
    use_scattering_router: bool = True
    scattering_scale: float = 0.1
    scattering_scale_warmup_steps: int = 1000
    use_birman_schwinger: bool = True
    epsilon: float = 1.0
    use_mourre: bool = True
    use_lap: bool = True
    schatten_threshold: float = 100.0
    precision_upgrade_threshold: float = 1e6
    use_bitnet: bool = True # Mandatory for d_model=100k
    use_symplectic: bool = False # Can be enabled for temporal dynamics
    symplectic_dt: float = 0.1

    # Semiseparable rank
    semiseparable_rank: Optional[int] = 32 # Keep low rank to survive OOM
