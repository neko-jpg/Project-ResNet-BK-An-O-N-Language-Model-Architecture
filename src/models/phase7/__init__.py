# src/models/phase7/__init__.py
"""
Phase 7: Hybrid Hyperbolic Attention Model

物理的直観:
Phase 7は双曲空間アテンションとSSMを組み合わせたハイブリッドモデルです。
- ローカルコンテキスト: 双曲空間での階層的関係を捉える（木構造、階層構造）
- グローバルコンテキスト: SSMによる効率的な長距離依存性（O(N)計算量）

Components:
- HyperbolicMultiHeadAttention: 双曲空間でのマルチヘッドアテンション
- HybridHyperbolicAttention: ローカル双曲アテンション + グローバルSSM
- Phase7IntegratedModel: HTT埋め込み + ハイブリッドアテンション統合モデル
- Phase7Config: Phase 7モデルの設定クラス

Note: integrated_model is imported lazily to avoid circular imports
"""

from .hyperbolic_attention import HyperbolicMultiHeadAttention, SingleHeadHyperbolicAttention

# Lazy import to avoid circular dependency with resnet_bk
def __getattr__(name):
    if name == "HybridHyperbolicAttention":
        from .hybrid_attention import HybridHyperbolicAttention
        return HybridHyperbolicAttention
    elif name == "Phase7IntegratedModel":
        from .integrated_model import Phase7IntegratedModel
        return Phase7IntegratedModel
    elif name == "Phase7Config":
        from .integrated_model import Phase7Config
        return Phase7Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HybridHyperbolicAttention",
    "HyperbolicMultiHeadAttention",
    "SingleHeadHyperbolicAttention",
    "Phase7IntegratedModel",
    "Phase7Config",
]
