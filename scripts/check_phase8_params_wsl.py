#!/usr/bin/env python3
"""
Phase 8 Parameter Count Verification Script

このスクリプトは以下を確認します:
1. モデルの総パラメータ数（圧縮前）
2. Low-Rank圧縮後の実効パラメータ数
3. 圧縮率の計算

Usage:
    python scripts/check_phase8_params_wsl.py --config configs/phase8_max_push.yaml
"""
import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase8.config import Phase8Config
from src.models.phase8.integrated_model import Phase8IntegratedModel


def count_parameters(model: nn.Module) -> dict:
    """
    モデルのパラメータ数をカウント
    
    Returns:
        dict: {
            'total': 総パラメータ数,
            'trainable': 訓練可能パラメータ数,
            'non_trainable': 訓練不可パラメータ数
        }
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def estimate_compressed_params(config: Phase8Config, total_params: int) -> dict:
    """
    Low-Rank圧縮後の実効パラメータ数を推定
    
    Args:
        config: Phase8Config
        total_params: 圧縮前の総パラメータ数
    
    Returns:
        dict: {
            'embedding_compression': Embedding圧縮率,
            'ffn_compression': FFN圧縮率,
            'estimated_compressed': 推定圧縮後パラメータ数,
            'compression_ratio': 圧縮率
        }
    """
    # Embedding圧縮: d_model → d_model/4 (75% reduction)
    embedding_compression = 0.25 if config.low_rank_embedding else 1.0
    
    # FFN圧縮: d_model → d_model/8 (87.5% reduction)
    ffn_compression = 0.125 if config.low_rank_ffn else 1.0
    
    # 推定: Embeddingが約10%, FFNが約70%, Attentionが約20%
    embedding_params = total_params * 0.10
    ffn_params = total_params * 0.70
    attention_params = total_params * 0.20
    
    compressed_embedding = embedding_params * embedding_compression
    compressed_ffn = ffn_params * ffn_compression
    compressed_attention = attention_params  # Attentionは圧縮しない
    
    estimated_compressed = compressed_embedding + compressed_ffn + compressed_attention
    compression_ratio = estimated_compressed / total_params
    
    return {
        'embedding_compression': embedding_compression,
        'ffn_compression': ffn_compression,
        'estimated_compressed': int(estimated_compressed),
        'compression_ratio': compression_ratio
    }


def format_number(num: int) -> str:
    """数値を読みやすい形式にフォーマット"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def main():
    parser = argparse.ArgumentParser(description='Phase 8 Parameter Count Verification')
    parser.add_argument('--config', type=str, default='configs/phase8_max_push.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 8 Parameter Count Verification")
    print("=" * 80)
    print()
    
    # Load config
    print(f"📄 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Phase8Config - only use model and optimization configs
    model_config = config_dict.get('model', {})
    optimization_config = config_dict.get('optimization', {})
    
    # Merge only model and optimization configs (skip training-specific params)
    merged_config = {**model_config, **optimization_config}
    
    # Filter out parameters that don't exist in Phase8Config
    # Get valid Phase8Config parameters
    from dataclasses import fields
    valid_params = {f.name for f in fields(Phase8Config)}
    
    # Filter merged_config to only include valid parameters
    filtered_config = {k: v for k, v in merged_config.items() if k in valid_params}
    
    # Create config object
    config = Phase8Config(**filtered_config)
    
    print(f"✓ Config loaded")
    print()
    
    # Display key parameters
    print("📊 Model Configuration:")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_layers: {config.n_layers}")
    print(f"  - num_heads: {config.num_heads}")
    print(f"  - n_seq: {config.n_seq}")
    print(f"  - vocab_size: {config.vocab_size}")
    print()
    
    print("🔧 Compression Settings:")
    print(f"  - Low-Rank Embedding: {config.low_rank_embedding}")
    print(f"  - Low-Rank FFN: {config.low_rank_ffn}")
    print()
    
    # Create model
    print("🏗️  Creating model...")
    try:
        model = Phase8IntegratedModel(config)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Count parameters
    print("🔢 Counting parameters...")
    param_counts = count_parameters(model)
    
    print()
    print("=" * 80)
    print("📈 PARAMETER COUNT RESULTS")
    print("=" * 80)
    print()
    
    print("🔹 Before Compression:")
    print(f"  Total Parameters:      {param_counts['total']:,} ({format_number(param_counts['total'])})")
    print(f"  Trainable Parameters:  {param_counts['trainable']:,} ({format_number(param_counts['trainable'])})")
    print(f"  Non-trainable:         {param_counts['non_trainable']:,} ({format_number(param_counts['non_trainable'])})")
    print()
    
    # Estimate compressed parameters
    compression_info = estimate_compressed_params(config, param_counts['total'])
    
    print("🔹 After Low-Rank Compression:")
    print(f"  Embedding Compression: {compression_info['embedding_compression']:.1%} (retain)")
    print(f"  FFN Compression:       {compression_info['ffn_compression']:.1%} (retain)")
    print(f"  Estimated Compressed:  {compression_info['estimated_compressed']:,} ({format_number(compression_info['estimated_compressed'])})")
    print(f"  Compression Ratio:     {compression_info['compression_ratio']:.1%}")
    print(f"  Reduction:             {(1 - compression_info['compression_ratio']) * 100:.1f}%")
    print()
    
    # Check if target is met
    target_before = 3_000_000_000  # 3B
    target_after = 30_000_000      # 30M
    
    print("=" * 80)
    print("🎯 TARGET VERIFICATION")
    print("=" * 80)
    print()
    
    before_ok = param_counts['total'] >= target_before
    after_ok = compression_info['estimated_compressed'] <= target_after * 2  # Allow 2x margin
    
    print(f"Target Before Compression: {format_number(target_before)}")
    print(f"Actual Before Compression: {format_number(param_counts['total'])}")
    print(f"Status: {'✅ PASS' if before_ok else '❌ FAIL'}")
    print()
    
    print(f"Target After Compression:  {format_number(target_after)}")
    print(f"Actual After Compression:  {format_number(compression_info['estimated_compressed'])}")
    print(f"Status: {'✅ PASS' if after_ok else '⚠️  WARNING (higher than target)'}")
    print()
    
    if before_ok and after_ok:
        print("=" * 80)
        print("🎉 SUCCESS: All targets met!")
        print("=" * 80)
    elif before_ok:
        print("=" * 80)
        print("⚠️  PARTIAL: Before compression target met, but after compression is higher than ideal")
        print("=" * 80)
    else:
        print("=" * 80)
        print("❌ FAIL: Targets not met. Consider increasing d_model or n_layers.")
        print("=" * 80)
    
    print()
    
    # Memory estimation
    print("=" * 80)
    print("💾 MEMORY ESTIMATION (FP16)")
    print("=" * 80)
    print()
    
    model_memory_gb = (param_counts['total'] * 2) / (1024 ** 3)  # FP16 = 2 bytes
    print(f"Model Memory:     {model_memory_gb:.2f} GB")
    print(f"Optimizer (8bit): {model_memory_gb * 0.5:.2f} GB (estimated)")
    print(f"Activations:      ~2.0 GB (with gradient checkpointing)")
    print(f"Peak VRAM:        ~{model_memory_gb + model_memory_gb * 0.5 + 2:.2f} GB")
    print()
    
    if model_memory_gb + model_memory_gb * 0.5 + 2 <= 8:
        print("✅ Should fit in 8GB VRAM")
    else:
        print("⚠️  May exceed 8GB VRAM - consider reducing batch size or using more aggressive optimizations")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
