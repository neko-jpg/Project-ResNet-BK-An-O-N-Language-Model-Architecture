"""
Phase 1 Model Conversion Utilities

このモジュールは、既存のMUSEモデルをPhase 1モデルに変換するユーティリティを提供します。
Embedding層のHTT変換、AR-SSM初期化、モデル重みの移行などを行います。

Requirements:
    - 4.1: 既存インフラとの統合
    - 4.4: 後方互換性の実装

Author: Project MUSE Team
"""

from typing import Optional, Dict, Any, Tuple
import warnings
import math

import torch
import torch.nn as nn

from .config import Phase1Config
from .ar_ssm_layer import AdaptiveRankSemiseparableLayer
from .htt_embedding import HolographicTTEmbedding, create_htt_embedding
from ..semiseparable_matrix import SemiseparableMatrix


def initialize_htt_from_embedding(
    htt_embedding: HolographicTTEmbedding,
    standard_embedding: nn.Embedding,
    method: str = "svd",
) -> HolographicTTEmbedding:
    """
    標準EmbeddingからHTT Embeddingの重みを初期化
    
    既存の訓練済みEmbedding重みを使用して、HTT Embeddingを初期化します。
    これにより、ゼロから訓練するよりも高速に収束します。
    
    Args:
        htt_embedding: 初期化対象のHolographicTTEmbedding
        standard_embedding: 既存のnn.Embedding（訓練済み）
        method: 初期化方法（"svd", "random", "mean"）
    
    Returns:
        初期化されたHolographicTTEmbedding
    
    Methods:
        - "svd": SVD分解を使用して低ランク近似
        - "random": ランダム初期化（デフォルトと同じ）
        - "mean": 平均値で初期化
    
    Example:
        >>> standard_emb = nn.Embedding(50000, 1024)
        >>> # ... train standard_emb ...
        >>> htt_emb = HolographicTTEmbedding(50000, 1024, rank=16)
        >>> htt_emb = initialize_htt_from_embedding(htt_emb, standard_emb, method="svd")
    
    Requirements: 4.1, 4.4
    """
    if method == "random":
        # Keep default random initialization
        return htt_embedding
    
    elif method == "mean":
        # Initialize with mean of standard embedding
        with torch.no_grad():
            mean_embedding = standard_embedding.weight.mean(dim=0)  # (d_model,)
            
            # Distribute mean across TT cores
            # This is a simple heuristic initialization
            htt_embedding.core1.data.fill_(0.0)
            htt_embedding.core2.data.fill_(0.0)
            
            # Set first rank dimension to capture mean
            htt_embedding.core1.data[:, 0, 0, :] = mean_embedding[:htt_embedding.d1].unsqueeze(0) * 0.1
            htt_embedding.core2.data[:, 0, 0, :] = mean_embedding[:htt_embedding.d2].unsqueeze(0) * 0.1
    
    elif method == "svd":
        # Use SVD to initialize TT cores
        # This provides a better low-rank approximation
        with torch.no_grad():
            # Get embedding matrix: (vocab_size, d_model)
            E = standard_embedding.weight.data
            vocab_size, d_model = E.shape
            
            # Reshape to match TT factorization
            # E: (vocab_size, d_model) → (v1, v2, d1, d2)
            v1, v2 = htt_embedding.v1, htt_embedding.v2
            d1, d2 = htt_embedding.d1, htt_embedding.d2
            rank = htt_embedding.rank
            
            # Pad if necessary
            E_padded = torch.zeros(v1 * v2, d1 * d2, device=E.device, dtype=E.dtype)
            E_padded[:vocab_size, :d_model] = E
            
            # Reshape to 4D tensor
            E_4d = E_padded.reshape(v1, v2, d1, d2)
            
            # Perform Tucker decomposition approximation
            # We'll use a simplified approach: factorize along vocab and dim separately
            
            # Factorize vocabulary dimension: (v1, v2) → (v1, r) × (r, v2)
            E_vocab = E_4d.reshape(v1, v2 * d1 * d2)
            U_vocab, S_vocab, Vt_vocab = torch.linalg.svd(E_vocab, full_matrices=False)
            
            # Keep top 'rank' components
            U_vocab_r = U_vocab[:, :rank]  # (v1, rank)
            S_vocab_r = S_vocab[:rank]  # (rank,)
            Vt_vocab_r = Vt_vocab[:rank, :]  # (rank, v2*d1*d2)
            
            # Factorize dimension: (d1, d2) → (d1, r) × (r, d2)
            E_dim = E_4d.reshape(v1 * v2 * d1, d2)
            U_dim, S_dim, Vt_dim = torch.linalg.svd(E_dim, full_matrices=False)
            
            # Keep top 'rank' components
            U_dim_r = U_dim[:, :rank]  # (v1*v2*d1, rank)
            S_dim_r = S_dim[:rank]  # (rank,)
            Vt_dim_r = Vt_dim[:rank, :]  # (rank, d2)
            
            # Initialize core1: (v1, 1, rank, d1)
            # Use vocabulary factorization
            core1_init = U_vocab_r.unsqueeze(1).unsqueeze(-1)  # (v1, 1, rank, 1)
            core1_init = core1_init.expand(v1, 1, rank, d1)
            core1_init = core1_init * (S_vocab_r.sqrt().view(1, 1, rank, 1) / math.sqrt(d1))
            
            # Initialize core2: (v2, rank, 1, d2)
            # Use dimension factorization
            Vt_vocab_r_reshaped = Vt_vocab_r.reshape(rank, v2, d1, d2)
            core2_init = Vt_vocab_r_reshaped.permute(1, 0, 2, 3).mean(dim=2, keepdim=True)  # (v2, rank, 1, d2)
            core2_init = core2_init * (S_dim_r.sqrt().view(1, rank, 1, 1) / math.sqrt(v2))
            
            # Assign to HTT embedding
            htt_embedding.core1.data.copy_(core1_init)
            htt_embedding.core2.data.copy_(core2_init)
            
            # Scale down to avoid initial instability
            htt_embedding.core1.data *= 0.1
            htt_embedding.core2.data *= 0.1
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return htt_embedding


def initialize_ar_ssm_from_semiseparable(
    ar_ssm: AdaptiveRankSemiseparableLayer,
    semisep: SemiseparableMatrix,
) -> AdaptiveRankSemiseparableLayer:
    """
    既存のSemiseparableMatrixからAR-SSMレイヤーを初期化
    
    既存の半可分行列の低ランク因子を使用して、AR-SSMレイヤーを初期化します。
    
    Args:
        ar_ssm: 初期化対象のAdaptiveRankSemiseparableLayer
        semisep: 既存のSemiseparableMatrix
    
    Returns:
        初期化されたAdaptiveRankSemiseparableLayer
    
    Example:
        >>> semisep = SemiseparableMatrix(n_seq=1024, rank=16)
        >>> ar_ssm = AdaptiveRankSemiseparableLayer(d_model=512, max_rank=32)
        >>> ar_ssm = initialize_ar_ssm_from_semiseparable(ar_ssm, semisep)
    
    Requirements: 4.1, 4.4
    """
    with torch.no_grad():
        # Get low-rank factors from semiseparable matrix
        if semisep.U is not None and semisep.V is not None:
            # semisep.U, semisep.V: (n_seq, rank)
            rank_min = min(ar_ssm.max_rank, semisep.rank)
            
            # Compute average U and V vectors
            u_avg = semisep.U.mean(dim=0)[:rank_min]  # (rank_min,)
            v_avg = semisep.V.mean(dim=0)[:rank_min]  # (rank_min,)
            
            # Initialize U_proj and V_proj weights
            # U_proj.weight: (max_rank, d_model)
            # V_proj.weight: (max_rank, d_model)
            
            # Strategy: Distribute low-rank factors across d_model dimensions
            for i in range(rank_min):
                # Initialize with scaled identity-like pattern
                ar_ssm.U_proj.weight.data[i, :] = u_avg[i] / math.sqrt(ar_ssm.d_model)
                ar_ssm.V_proj.weight.data[i, :] = v_avg[i] / math.sqrt(ar_ssm.d_model)
        
        # Initialize T_conv from tridiagonal structure
        if hasattr(semisep, 'main_diag') and semisep.main_diag is not None:
            # Use main diagonal to initialize conv weights
            diag_avg = semisep.main_diag.mean()
            ar_ssm.T_conv.weight.data.fill_(diag_avg / 3.0)  # Divide by kernel_size
    
    return ar_ssm


def convert_embedding_to_htt(
    model: nn.Module,
    config: Phase1Config,
    embedding_attr: str = "token_embedding",
    initialize_from_weights: bool = True,
    initialization_method: str = "svd",
) -> nn.Module:
    """
    モデル内の指定されたEmbedding層をHTTに変換
    
    Args:
        model: 対象モデル
        config: Phase1Config
        embedding_attr: Embedding層の属性名
        initialize_from_weights: 既存の重みから初期化するか
        initialization_method: 初期化方法（"svd", "random", "mean"）
    
    Returns:
        Modified model (in-place)
    
    Example:
        >>> model = LanguageModel(vocab_size=50000, d_model=1024)
        >>> # ... train model ...
        >>> config = Phase1Config(htt_rank=16)
        >>> model = convert_embedding_to_htt(model, config, initialize_from_weights=True)
    
    Requirements: 4.1, 4.4
    """
    if not hasattr(model, embedding_attr):
        raise ValueError(f"Model does not have attribute '{embedding_attr}'")
    
    old_embedding = getattr(model, embedding_attr)
    
    if not isinstance(old_embedding, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding, got {type(old_embedding)}")
    
    vocab_size = old_embedding.num_embeddings
    d_model = old_embedding.embedding_dim
    
    # Create HTT embedding
    new_embedding = create_htt_embedding(vocab_size, d_model, config)
    
    # Initialize from existing weights if requested
    if initialize_from_weights:
        new_embedding = initialize_htt_from_embedding(
            new_embedding,
            old_embedding,
            method=initialization_method,
        )
        
        print(f"Initialized HTT embedding from existing weights using '{initialization_method}' method")
    
    # Replace embedding
    setattr(model, embedding_attr, new_embedding)
    
    # Print compression info
    standard_params, tt_params = new_embedding.get_parameter_counts()
    compression_ratio = new_embedding.get_compression_ratio()
    print(f"Converted {embedding_attr}: {standard_params:,} → {tt_params:,} params "
          f"({compression_ratio:.2%} compression)")
    
    return model


def convert_all_embeddings_to_htt(
    model: nn.Module,
    config: Phase1Config,
    initialize_from_weights: bool = True,
    initialization_method: str = "svd",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    モデル内のすべてのEmbedding層をHTTに変換
    
    Args:
        model: 対象モデル
        config: Phase1Config
        initialize_from_weights: 既存の重みから初期化するか
        initialization_method: 初期化方法
    
    Returns:
        (modified_model, conversion_info)
        conversion_info: 変換情報の辞書
    
    Example:
        >>> model = LanguageModel(vocab_size=50000, d_model=1024)
        >>> config = Phase1Config(htt_rank=16)
        >>> model, info = convert_all_embeddings_to_htt(model, config)
        >>> print(f"Converted {info['num_converted']} embeddings")
    
    Requirements: 4.1, 4.4
    """
    conversion_info = {
        'num_converted': 0,
        'total_params_before': 0,
        'total_params_after': 0,
        'conversions': [],
    }
    
    # Find all embeddings
    embeddings_to_convert = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embeddings_to_convert.append((name, module))
    
    # Convert each embedding
    for name, old_embedding in embeddings_to_convert:
        # Get parent module and attribute name
        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
        attr_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        # Create HTT embedding
        vocab_size = old_embedding.num_embeddings
        d_model = old_embedding.embedding_dim
        new_embedding = create_htt_embedding(vocab_size, d_model, config)
        
        # Initialize from existing weights if requested
        if initialize_from_weights:
            new_embedding = initialize_htt_from_embedding(
                new_embedding,
                old_embedding,
                method=initialization_method,
            )
        
        # Replace embedding
        setattr(parent, attr_name, new_embedding)
        
        # Track conversion info
        standard_params, tt_params = new_embedding.get_parameter_counts()
        conversion_info['num_converted'] += 1
        conversion_info['total_params_before'] += standard_params
        conversion_info['total_params_after'] += tt_params
        conversion_info['conversions'].append({
            'name': name,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'params_before': standard_params,
            'params_after': tt_params,
            'compression_ratio': new_embedding.get_compression_ratio(),
        })
        
        print(f"Converted {name}: {standard_params:,} → {tt_params:,} params")
    
    # Calculate overall compression
    if conversion_info['total_params_before'] > 0:
        overall_compression = conversion_info['total_params_after'] / conversion_info['total_params_before']
        conversion_info['overall_compression_ratio'] = overall_compression
        conversion_info['overall_compression_percentage'] = (1 - overall_compression) * 100
    
    print(f"\nTotal: Converted {conversion_info['num_converted']} embeddings")
    print(f"Parameters: {conversion_info['total_params_before']:,} → "
          f"{conversion_info['total_params_after']:,} "
          f"({conversion_info.get('overall_compression_percentage', 0):.1f}% reduction)")
    
    return model, conversion_info


def add_ar_ssm_layers(
    model: nn.Module,
    config: Phase1Config,
    layer_positions: Optional[list] = None,
    d_model: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    モデルにAR-SSMレイヤーを追加
    
    注意: この関数は汎用的な実装の雛形です。
    実際のモデルアーキテクチャに応じて、適切な位置にAR-SSMを挿入する
    カスタム実装が必要です。
    
    Args:
        model: 対象モデル
        config: Phase1Config
        layer_positions: AR-SSMを追加するレイヤー位置（Noneの場合は自動検出）
        d_model: モデル次元（Noneの場合は自動検出）
    
    Returns:
        (modified_model, addition_info)
    
    Example:
        >>> model = LanguageModel(vocab_size=50000, d_model=1024, n_layers=12)
        >>> config = Phase1Config()
        >>> model, info = add_ar_ssm_layers(model, config, d_model=1024)
    
    Requirements: 4.1, 4.4
    """
    warnings.warn(
        "add_ar_ssm_layers is a generic template. "
        "Custom implementation required for specific model architectures. "
        "Consider using Phase1IntegratedModel or manual integration instead.",
        UserWarning
    )
    
    addition_info = {
        'num_added': 0,
        'positions': [],
    }
    
    # This is a placeholder implementation
    # In practice, you would need to:
    # 1. Identify appropriate insertion points in the model
    # 2. Create AR-SSM layers with correct dimensions
    # 3. Insert them into the model architecture
    # 4. Optionally initialize from existing SemiseparableMatrix layers
    
    # Example: Look for SemiseparableMatrix layers and add AR-SSM alongside
    for name, module in model.named_modules():
        if isinstance(module, SemiseparableMatrix):
            # Found a semiseparable matrix layer
            # Could add AR-SSM here, but need to know model architecture
            addition_info['positions'].append(name)
    
    if addition_info['positions']:
        print(f"Found {len(addition_info['positions'])} potential positions for AR-SSM:")
        for pos in addition_info['positions']:
            print(f"  - {pos}")
        print("\nManual integration recommended for production use.")
    else:
        print("No SemiseparableMatrix layers found. Manual integration required.")
    
    return model, addition_info


def verify_conversion(
    original_model: nn.Module,
    converted_model: nn.Module,
    test_input: torch.Tensor,
    tolerance: float = 1e-2,
) -> Dict[str, Any]:
    """
    変換されたモデルの出力を元のモデルと比較して検証
    
    Args:
        original_model: 元のモデル
        converted_model: 変換後のモデル
        test_input: テスト入力
        tolerance: 許容誤差
    
    Returns:
        検証結果の辞書
    
    Example:
        >>> original = LanguageModel(vocab_size=1000, d_model=128)
        >>> converted = convert_model_to_phase1(original)
        >>> test_input = torch.randint(0, 1000, (2, 10))
        >>> result = verify_conversion(original, converted, test_input)
        >>> assert result['outputs_close'], "Conversion failed verification"
    
    Requirements: 4.1, 4.4
    """
    original_model.eval()
    converted_model.eval()
    
    with torch.no_grad():
        # Forward pass
        try:
            original_output = original_model(test_input)
            converted_output = converted_model(test_input)
            
            # Handle tuple outputs (e.g., (logits, hidden_states))
            if isinstance(original_output, tuple):
                original_output = original_output[0]
            if isinstance(converted_output, tuple):
                converted_output = converted_output[0]
            
            # Compute difference
            diff = (original_output - converted_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            outputs_close = max_diff < tolerance
            
            result = {
                'outputs_close': outputs_close,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'tolerance': tolerance,
                'original_shape': tuple(original_output.shape),
                'converted_shape': tuple(converted_output.shape),
                'shapes_match': original_output.shape == converted_output.shape,
            }
            
            if outputs_close:
                print(f"✓ Conversion verified: max_diff={max_diff:.6f} < tolerance={tolerance}")
            else:
                print(f"✗ Conversion verification failed: max_diff={max_diff:.6f} >= tolerance={tolerance}")
                print(f"  Mean diff: {mean_diff:.6f}")
            
            return result
            
        except Exception as e:
            print(f"✗ Conversion verification failed with error: {e}")
            return {
                'outputs_close': False,
                'error': str(e),
            }


def get_conversion_summary(model: nn.Module) -> Dict[str, Any]:
    """
    モデルのPhase 1変換状況のサマリーを取得
    
    Args:
        model: 対象モデル
    
    Returns:
        変換状況のサマリー辞書
    
    Example:
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024)
        >>> summary = get_conversion_summary(model)
        >>> print(f"HTT embeddings: {summary['num_htt_embeddings']}")
    
    Requirements: 4.1
    """
    summary = {
        'num_htt_embeddings': 0,
        'num_standard_embeddings': 0,
        'num_ar_ssm_layers': 0,
        'num_semiseparable_layers': 0,
        'num_lns_linear_layers': 0,
        'num_standard_linear_layers': 0,
        'total_params': 0,
        'htt_params': 0,
        'standard_embedding_params': 0,
    }
    
    for module in model.modules():
        if isinstance(module, HolographicTTEmbedding):
            summary['num_htt_embeddings'] += 1
            _, tt_params = module.get_parameter_counts()
            summary['htt_params'] += tt_params
        elif isinstance(module, nn.Embedding):
            summary['num_standard_embeddings'] += 1
            summary['standard_embedding_params'] += module.num_embeddings * module.embedding_dim
        elif isinstance(module, AdaptiveRankSemiseparableLayer):
            summary['num_ar_ssm_layers'] += 1
        elif isinstance(module, SemiseparableMatrix):
            summary['num_semiseparable_layers'] += 1
        elif hasattr(module, '__class__') and 'LNSLinear' in module.__class__.__name__:
            summary['num_lns_linear_layers'] += 1
        elif isinstance(module, nn.Linear):
            summary['num_standard_linear_layers'] += 1
    
    # Total parameters
    summary['total_params'] = sum(p.numel() for p in model.parameters())
    
    # Conversion percentage
    if summary['num_htt_embeddings'] + summary['num_standard_embeddings'] > 0:
        summary['embedding_conversion_percentage'] = (
            summary['num_htt_embeddings'] / 
            (summary['num_htt_embeddings'] + summary['num_standard_embeddings']) * 100
        )
    else:
        summary['embedding_conversion_percentage'] = 0.0
    
    return summary


def print_conversion_summary(model: nn.Module):
    """
    モデルのPhase 1変換状況を出力
    
    Args:
        model: 対象モデル
    
    Example:
        >>> model = create_phase1_model(vocab_size=50000, d_model=1024)
        >>> print_conversion_summary(model)
    """
    summary = get_conversion_summary(model)
    
    print("=" * 60)
    print("Phase 1 Conversion Summary")
    print("=" * 60)
    print()
    print("Embeddings:")
    print(f"  HTT Embeddings: {summary['num_htt_embeddings']}")
    print(f"  Standard Embeddings: {summary['num_standard_embeddings']}")
    print(f"  Conversion: {summary['embedding_conversion_percentage']:.1f}%")
    print()
    print("Layers:")
    print(f"  AR-SSM Layers: {summary['num_ar_ssm_layers']}")
    print(f"  Semiseparable Layers: {summary['num_semiseparable_layers']}")
    print(f"  LNS Linear Layers: {summary['num_lns_linear_layers']}")
    print(f"  Standard Linear Layers: {summary['num_standard_linear_layers']}")
    print()
    print("Parameters:")
    print(f"  Total: {summary['total_params']:,}")
    print(f"  HTT Embedding: {summary['htt_params']:,}")
    print(f"  Standard Embedding: {summary['standard_embedding_params']:,}")
    print()
