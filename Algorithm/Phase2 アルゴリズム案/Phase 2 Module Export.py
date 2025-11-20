"""
Project MUSE Phase 2: Breath of Life (生命の息吹)
Modules for dynamic time evolution, forgetting, and fractal initialization.
"""

from .non_hermitian import NonHermitianPotential, DissipativeBKLayer
from .hebbian import HebbianFastWeights, DynamicPotentialUpdater
from .zeta import ZetaInitializer, ZetaEmbedding

__all__ = [
    "NonHermitianPotential",
    "DissipativeBKLayer",
    "HebbianFastWeights",
    "DynamicPotentialUpdater",
    "ZetaInitializer",
    "ZetaEmbedding",
]