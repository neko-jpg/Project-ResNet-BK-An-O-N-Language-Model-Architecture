"""
Training Module
Contains training loops and optimization utilities.
"""

from .hybrid_koopman_trainer import HybridKoopmanTrainer
from .koopman_scheduler import KoopmanLossScheduler
from .physics_informed_trainer import PhysicsInformedTrainer
from .symplectic_optimizer import SymplecticSGD, SymplecticAdam, create_symplectic_optimizer
from .equilibrium_propagation import EquilibriumPropagationTrainer, HybridEquilibriumTrainer

# Step 7: System Integration and Data Efficiency
from .curriculum_learning import CurriculumLearningScheduler, DynamicDifficultyAdjuster, create_curriculum_trainer
from .active_learning import ActiveLearningSelector, ActiveLearningLoop, create_active_learning_trainer
from .data_augmentation import LanguageDataAugmenter, BackTranslationAugmenter, create_augmented_dataset
from .transfer_learning import TransferLearningPipeline, DomainAdaptationPipeline, create_transfer_learning_pipeline
from .gradient_caching import GradientCachingTrainer, AdaptiveGradientCachingTrainer, create_gradient_caching_trainer
from .difficulty_prediction import DifficultyPredictor, DifficultyPredictionTrainer, train_with_difficulty_prediction
from .dynamic_lr_scheduler import DynamicLRScheduler, CosineAnnealingWarmRestarts, OneCycleLR, create_dynamic_scheduler
from .distributed_optimizations import ZeROOptimizer, DistributedTrainer, GradientAccumulator, train_with_gradient_accumulation

# Mamba-Killer: Failure Recovery and Monitoring
from .stability_monitor import StabilityMonitor, StabilityMetrics
from .auto_recovery import AutoRecovery, RecoveryState
from .colab_timeout_handler import ColabTimeoutHandler

__all__ = [
    # Step 2: Koopman and Physics-Informed Learning
    'HybridKoopmanTrainer',
    'KoopmanLossScheduler',
    'PhysicsInformedTrainer',
    'SymplecticSGD',
    'SymplecticAdam',
    'create_symplectic_optimizer',
    'EquilibriumPropagationTrainer',
    'HybridEquilibriumTrainer',
    
    # Step 7: System Integration and Data Efficiency
    'CurriculumLearningScheduler',
    'DynamicDifficultyAdjuster',
    'create_curriculum_trainer',
    'ActiveLearningSelector',
    'ActiveLearningLoop',
    'create_active_learning_trainer',
    'LanguageDataAugmenter',
    'BackTranslationAugmenter',
    'create_augmented_dataset',
    'TransferLearningPipeline',
    'DomainAdaptationPipeline',
    'create_transfer_learning_pipeline',
    'GradientCachingTrainer',
    'AdaptiveGradientCachingTrainer',
    'create_gradient_caching_trainer',
    'DifficultyPredictor',
    'DifficultyPredictionTrainer',
    'train_with_difficulty_prediction',
    'DynamicLRScheduler',
    'CosineAnnealingWarmRestarts',
    'OneCycleLR',
    'create_dynamic_scheduler',
    'ZeROOptimizer',
    'DistributedTrainer',
    'GradientAccumulator',
    'train_with_gradient_accumulation',
    
    # Mamba-Killer: Failure Recovery and Monitoring
    'StabilityMonitor',
    'StabilityMetrics',
    'AutoRecovery',
    'RecoveryState',
    'ColabTimeoutHandler',
]
