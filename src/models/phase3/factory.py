"""
Model Factory for Phase 3: Physics Transcendence

このモジュールは、Phase 3モデルの生成と、Phase 2からのモデル変換機能を提供します。

Requirements:
    - Requirement 7.8: create_phase3_model Function
    - Requirement 7.9: convert_phase2_to_phase3 Function
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any, Union

from src.models.phase3.integrated_model import Phase3IntegratedModel
from src.models.phase3.config import Phase3Config
from src.models.phase3.complex_embedding import convert_phase2_embedding_to_complex

# Phase 2 model import (Assuming standard location)
# from src.models.phase2.integrated_model import Phase2IntegratedModel # Hypothetical

def create_phase3_model(
    config: Union[Phase3Config, Dict[str, Any]]
) -> Phase3IntegratedModel:
    """
    Task 21.1: create_phase3_model Function

    設定からPhase 3モデルを生成します。

    Args:
        config: Phase3Configオブジェクトまたは辞書

    Returns:
        model: 初期化されたPhase3IntegratedModel
    """
    if isinstance(config, dict):
        # 辞書からConfigオブジェクトを作成
        # 必須パラメータのチェックはdataclassが行う
        config = Phase3Config(**config)

    print(f"Creating Phase 3 Model with config: {config}")
    model = Phase3IntegratedModel(config)
    return model


def convert_phase2_to_phase3(
    phase2_model: nn.Module,
    phase3_config: Optional[Phase3Config] = None
) -> Phase3IntegratedModel:
    """
    Task 21.2: convert_phase2_to_phase3 Function

    Phase 2モデルの重みをPhase 3モデルに変換します。

    Conversion Logic:
        - Embedding: Phase 2の重みを実部にコピー、虚部はゼロ初期化
        - Layers:
            - Phase 2層のパラメータを可能な限り対応するPhase 3層にマップ
            - 構造が大きく異なるため、完全な変換は困難
            - ここではEmbeddingの変換と、一般的なパラメータの形状マッチングを行う

    Args:
        phase2_model: Phase 2モデルインスタンス
        phase3_config: Phase 3設定（Noneの場合はPhase 2から推定）

    Returns:
        phase3_model: 重みが転送されたPhase 3モデル
    """
    # Phase 3 Configの推定
    if phase3_config is None:
        # phase2_model.configが存在すると仮定
        if hasattr(phase2_model, 'config'):
            p2_conf = phase2_model.config
            phase3_config = Phase3Config(
                vocab_size=getattr(p2_conf, 'vocab_size', 50000),
                d_model=getattr(p2_conf, 'd_model', 512),
                n_layers=getattr(p2_conf, 'n_layers', 6),
                max_seq_len=getattr(p2_conf, 'max_seq_len', 2048)
            )
        else:
            raise ValueError("Cannot infer Phase 3 config from Phase 2 model. Please provide phase3_config.")

    # Phase 3モデルの作成
    phase3_model = create_phase3_model(phase3_config)

    # 1. Embedding Conversion
    # Phase 2 model is assumed to have 'embedding' attribute which is nn.Embedding or compatible
    if hasattr(phase2_model, 'embedding') and hasattr(phase3_model, 'embedding'):
        # Check if phase2 embedding is standard nn.Embedding
        p2_emb = phase2_model.embedding

        # If p2_emb is wrapped (e.g. ZetaEmbedding), access internal embedding
        if hasattr(p2_emb, 'embedding'):
             p2_emb = p2_emb.embedding

        if isinstance(p2_emb, nn.Embedding):
             # Use the conversion utility
             print("Converting Embedding layer...")
             # Note: convert_phase2_embedding_to_complex creates a NEW instance,
             # but phase3_model already has one. We should copy weights.
             with torch.no_grad():
                 phase3_model.embedding.token_embedding_real.weight.copy_(p2_emb.weight)
                 phase3_model.embedding.token_embedding_imag.weight.zero_()
                 print("Embedding weights transferred.")

    # 2. Layer Conversion (Heuristic)
    # Phase 3 layers are complex and different. We can't easily map MLP weights directly
    # unless we map Real->Real parts.
    # Phase 3 Block: Norm -> Hamiltonian(BKCore/MLP) -> Koopman -> Residual
    # Phase 2 Block: Norm -> Attention/SSM -> MLP -> Residual?

    # For now, we focus on transferring what's possible, e.g. Output Head
    # If phase2 has output head
    if hasattr(phase2_model, 'head') and hasattr(phase3_model, 'dialectic'):
        # Dialectic loop has generator_head
        p2_head = phase2_model.head
        p3_head = phase3_model.dialectic.generator_head

        if isinstance(p2_head, nn.Linear) and isinstance(p3_head, nn.Linear):
            if p2_head.weight.shape == p3_head.weight.shape:
                 with torch.no_grad():
                     p3_head.weight.copy_(p2_head.weight)
                     if p2_head.bias is not None and p3_head.bias is not None:
                         p3_head.bias.copy_(p2_head.bias)
                 print("Output Head weights transferred.")

    print("Model conversion completed (partial). Pre-training or fine-tuning is recommended.")
    return phase3_model

def get_preset_config(preset_name: str) -> Phase3Config:
    """プリセット設定の取得"""
    presets = {
        'small': Phase3Config(vocab_size=50000, d_model=256, n_layers=4, max_seq_len=1024),
        'base': Phase3Config(vocab_size=50000, d_model=512, n_layers=6, max_seq_len=2048),
        'large': Phase3Config(vocab_size=50000, d_model=1024, n_layers=12, max_seq_len=4096),
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(presets.keys())}")

    return presets[preset_name]
