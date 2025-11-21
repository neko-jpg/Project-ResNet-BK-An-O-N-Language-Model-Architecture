"""
Phase 3: Physics Transcendence

Phase 3は、Project MUSEの最終フェーズであり、以下の7つの核心技術を統合します：

1. Complex Dynamics (3.1) - 複素数ニューラルネットワーク
2. Hamiltonian Neural ODE (3.2) - エネルギー保存思考
3. MERA Routing (3.3) - 階層的情報集約
4. Symplectic Adjoint (3.4) - O(1)メモリ学習
5. Koopman Linearization (3.5) - 非線形系の線形化
6. Entropic Selection (3.6) - データ圧縮
7. Dialectic Loop (3.7) - 自己進化機構
"""

from .complex_tensor import ComplexTensor
from .complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm
from .complex_embedding import (
    ComplexEmbedding,
    convert_phase2_embedding_to_complex,
    analyze_complex_embedding_interference
)
from .stage1_model import (
    Phase3Stage1Block,
    Phase3Stage1Model,
    create_phase3_stage1_model,
    convert_phase2_to_complex,
    load_phase2_checkpoint_and_convert,
    compare_phase2_phase3_outputs,
)
from .hamiltonian import (
    HamiltonianFunction,
    symplectic_leapfrog_step,
    monitor_energy_conservation,
)

__all__ = [
    'ComplexTensor',
    'ComplexLinear',
    'ModReLU',
    'ComplexLayerNorm',
    'ComplexEmbedding',
    'convert_phase2_embedding_to_complex',
    'analyze_complex_embedding_interference',
    'Phase3Stage1Block',
    'Phase3Stage1Model',
    'create_phase3_stage1_model',
    'convert_phase2_to_complex',
    'load_phase2_checkpoint_and_convert',
    'compare_phase2_phase3_outputs',
    'HamiltonianFunction',
    'symplectic_leapfrog_step',
    'monitor_energy_conservation',
]
