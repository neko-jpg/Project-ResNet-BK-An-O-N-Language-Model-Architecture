"""
ResNet-BK Models Module
Contains all model architectures and components.
"""

from .bk_core import BKCoreFunction, get_tridiagonal_inverse_diagonal
from .moe import SparseMoELayer
from .resnet_bk import MoEResNetBKLayer, ResNetBKBlock, LanguageModel
from .configurable_resnet_bk import ConfigurableResNetBK
from .koopman_layer import (
    KoopmanResNetBKLayer,
    KoopmanResNetBKBlock,
    KoopmanLanguageModel
)
from .physics_informed_layer import PhysicsInformedBKLayer
from .adaptive_computation import (
    AdaptiveResNetBKBlock,
    ACTLanguageModel,
    ACTTrainer
)
from .semiseparable_matrix import (
    SemiseparableMatrix,
    SemiseparableCheckpointFunction,
    create_semiseparable_from_dense
)
from .mamba_baseline import (
    MambaLM,
    MambaConfig,
    MambaBlock,
    create_mamba_from_resnetbk_config
)

# Hugging Face integration (optional import)
try:
    from .hf_resnet_bk import (
        ResNetBKConfig,
        ResNetBKForCausalLM,
        create_resnet_bk_for_hf
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    ResNetBKConfig = None
    ResNetBKForCausalLM = None
    create_resnet_bk_for_hf = None

__all__ = [
    'BKCoreFunction',
    'get_tridiagonal_inverse_diagonal',
    'SparseMoELayer',
    'MoEResNetBKLayer',
    'ResNetBKBlock',
    'LanguageModel',
    'ConfigurableResNetBK',
    'KoopmanResNetBKLayer',
    'KoopmanResNetBKBlock',
    'KoopmanLanguageModel',
    'PhysicsInformedBKLayer',
    'AdaptiveResNetBKBlock',
    'ACTLanguageModel',
    'ACTTrainer',
    'SemiseparableMatrix',
    'SemiseparableCheckpointFunction',
    'create_semiseparable_from_dense',
    'MambaLM',
    'MambaConfig',
    'MambaBlock',
    'create_mamba_from_resnetbk_config',
    'ResNetBKConfig',
    'ResNetBKForCausalLM',
    'create_resnet_bk_for_hf',
    'HF_AVAILABLE',
]
