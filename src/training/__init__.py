"""
Training Module
Contains training loops and optimization utilities.
"""

from .hybrid_koopman_trainer import HybridKoopmanTrainer
from .koopman_scheduler import KoopmanLossScheduler
from .physics_informed_trainer import PhysicsInformedTrainer
from .symplectic_optimizer import SymplecticSGD, SymplecticAdam, create_symplectic_optimizer
from .equilibrium_propagation import EquilibriumPropagationTrainer, HybridEquilibriumTrainer

__all__ = [
    'HybridKoopmanTrainer',
    'KoopmanLossScheduler',
    'PhysicsInformedTrainer',
    'SymplecticSGD',
    'SymplecticAdam',
    'create_symplectic_optimizer',
    'EquilibriumPropagationTrainer',
    'HybridEquilibriumTrainer',
]
