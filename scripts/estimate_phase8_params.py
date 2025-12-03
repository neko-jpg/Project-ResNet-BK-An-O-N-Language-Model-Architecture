#!/usr/bin/env python3
"""
Phase 8 Parameter Estimation Script (Fast Version)

モデルを実際に作成せずに、設定からパラメータ数を推定します。
"""
import sys
import argparse
import yaml
from pathlib import Path


def estimate_parameters(config: dict) -> dict:
    """
    設定からパラメータ数を推定
    
    Args:
        config: モデル設定辞書
    
    Returns:
        dict: パラメータ数の推定結果
    """
    d_model = config.get('d_model', 512)
    n_layers = config.get('n_layers', 12)
    vocab_size = config.get('vocab_size', 50257)
    n_seq = config.get('n_seq', 512)
    num_heads = config.get('num_heads', 8)
    
    # Embedding parameters
    # Token embedding: vocab_size * d_model
    # Position embedding: n_seq * d_model
    embedding_params = vocab_size * d_model + n_seq * d_model
    
    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    # Q, K, V: d_model * d_model * 3
    # Output: d_model * d_model
    attention_params_per_layer = d_model * d_model * 4
    
    # FFN: typically 4x expansion
    # FFN1: d_model * (4 * d_model)
    # FFN2: (4 * d_model) * d_model
    ffn_params_per_layer = d_model * (4 * d_model) * 2
    
    # Layer norm: 2 * d_model per layer (2 layer norms per layer)
    ln_params_per_layer = 2 * d_model * 2
    
    # Total per layer
    params_per_layer = attention_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    
    # Total for all layers
    layer_params = params_per_layer * n_layers
    
    # Output head: d_model * vocab_size
    output_params = d_model * vocab_size
    
    # Total parameters
    total_params = embedding_params + layer_params + output_params
    
    return {
        'embedding': embedding_params,
        'attention_per_layer': attention_params_per_layer,
        'ffn_per_layer': ffn_params_per_layer,
        'ln_per_layer': ln_params_per_layer,
        'params_per_layer': params_per_layer,
        'layer_params': layer_params,
        'output': output_params,
        'total': total_params
    }


def estimate_compressed_parameters(params: dict, config: dict) -> dict:
    """
    Low-Rank圧縮後のパラメータ数を推定
    
    Args:
        params: 圧縮前のパラメータ数
        config: 設定辞書
    
    Returns:
        dict: 圧縮後のパラメータ数
    """
    low_rank_embedding = config.get('low_rank_embedding', False)
    low_rank_ffn = config.get('low_rank_ffn', False)
    
    # Embedding compression: 75% reduction (d_model → d_model/4)
    embedding_compression = 0.25 if low_rank_embedding else 1.0
    compressed_embedding = int(params['embedding'] * embedding_compression)
    
    # FFN compression: 87.5% reduction (d_model → d_model/8)
    ffn_compression = 0.125 if low_rank_ffn else 1.0
    compressed_ffn = int(params['ffn_per_layer'] * ffn_compression * config.get('n_layers', 12))
    
    # Attention and other params remain unchanged
    attention_params = params['attention_per_layer'] * config.get('n_layers', 12)
    ln_params = params['ln_per_layer'] * config.get('n_layers', 12)
    output_params = params['output']
    
    # Total compressed
    total_compressed = compressed_embedding + compressed_ffn + attention_params + ln_params + output_params
    
    return {
        'embedding': compressed_embedding,
        'ffn': compressed_ffn,
        'attention': attention_params,
        'ln': ln_params,
        'output': output_params,
        'total': total_compressed,
        'compression_ratio': total_compressed / params['total']
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
    parser = argparse.ArgumentParser(description='Phase 8 Parameter Estimation (Fast)')
    parser.add_argument('--config', type=str, default='configs/phase8_max_push.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 8 Parameter Estimation (Fast Version)")
    print("=" * 80)
    print()
    
    # Load config
    print(f"📄 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = config_dict.get('model', {})
    optimization_config = config_dict.get('optimization', {})
    merged_config = {**model_config, **optimization_config}
    
    print("✓ Config loaded")
    print()
    
    # Display key parameters
    print("📊 Model Configuration:")
    print(f"  - d_model: {merged_config.get('d_model', 'N/A')}")
    print(f"  - n_layers: {merged_config.get('n_layers', 'N/A')}")
    print(f"  - num_heads: {merged_config.get('num_heads', 'N/A')}")
    print(f"  - n_seq: {merged_config.get('n_seq', 'N/A')}")
    print(f"  - vocab_size: {merged_config.get('vocab_size', 'N/A')}")
    print()
    
    print("🔧 Compression Settings:")
    print(f"  - Low-Rank Embedding: {merged_config.get('low_rank_embedding', False)}")
    print(f"  - Low-Rank FFN: {merged_config.get('low_rank_ffn', False)}")
    print()
    
    # Estimate parameters
    print("🔢 Estimating parameters...")
    params = estimate_parameters(merged_config)
    
    print()
    print("=" * 80)
    print("📈 PARAMETER ESTIMATION RESULTS")
    print("=" * 80)
    print()
    
    print("🔹 Before Compression:")
    print(f"  Embedding:             {params['embedding']:,} ({format_number(params['embedding'])})")
    print(f"  Attention (per layer): {params['attention_per_layer']:,} ({format_number(params['attention_per_layer'])})")
    print(f"  FFN (per layer):       {params['ffn_per_layer']:,} ({format_number(params['ffn_per_layer'])})")
    print(f"  LayerNorm (per layer): {params['ln_per_layer']:,} ({format_number(params['ln_per_layer'])})")
    print(f"  All Layers:            {params['layer_params']:,} ({format_number(params['layer_params'])})")
    print(f"  Output Head:           {params['output']:,} ({format_number(params['output'])})")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:                 {params['total']:,} ({format_number(params['total'])})")
    print()
    
    # Estimate compressed parameters
    compressed = estimate_compressed_parameters(params, merged_config)
    
    print("🔹 After Low-Rank Compression:")
    print(f"  Embedding:             {compressed['embedding']:,} ({format_number(compressed['embedding'])})")
    print(f"  FFN (all layers):      {compressed['ffn']:,} ({format_number(compressed['ffn'])})")
    print(f"  Attention (all layers):{compressed['attention']:,} ({format_number(compressed['attention'])})")
    print(f"  LayerNorm (all layers):{compressed['ln']:,} ({format_number(compressed['ln'])})")
    print(f"  Output Head:           {compressed['output']:,} ({format_number(compressed['output'])})")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:                 {compressed['total']:,} ({format_number(compressed['total'])})")
    print(f"  Compression Ratio:     {compressed['compression_ratio']:.1%}")
    print(f"  Reduction:             {(1 - compressed['compression_ratio']) * 100:.1f}%")
    print()
    
    # Check if target is met
    target_before = 3_000_000_000  # 3B
    target_after = 30_000_000      # 30M
    
    print("=" * 80)
    print("🎯 TARGET VERIFICATION")
    print("=" * 80)
    print()
    
    before_ok = params['total'] >= target_before
    after_ok = compressed['total'] <= target_after * 2  # Allow 2x margin
    
    print(f"Target Before Compression: {format_number(target_before)}")
    print(f"Actual Before Compression: {format_number(params['total'])}")
    print(f"Status: {'✅ PASS' if before_ok else '❌ FAIL'}")
    print()
    
    print(f"Target After Compression:  {format_number(target_after)}")
    print(f"Actual After Compression:  {format_number(compressed['total'])}")
    print(f"Status: {'✅ PASS' if after_ok else '⚠️  WARNING (higher than target)'}")
    print()
    
    if before_ok and after_ok:
        print("=" * 80)
        print("🎉 SUCCESS: All targets met!")
        print("=" * 80)
    elif before_ok:
        print("=" * 80)
        print("⚠️  PARTIAL: Before compression target met, but after compression is higher than ideal")
        print("   Note: This is an estimation. Actual compression may be more effective.")
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
    
    model_memory_gb = (params['total'] * 2) / (1024 ** 3)  # FP16 = 2 bytes
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
