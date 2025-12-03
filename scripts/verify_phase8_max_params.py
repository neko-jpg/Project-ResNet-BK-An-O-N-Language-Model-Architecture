#!/usr/bin/env python3
"""
Phase 8 Max Parameters Verification Script
phase8_max_push.yamlの設定で本当に3億パラメータで、圧縮後30Mになっているか確認

Usage:
    wsl -d ubuntu -- bash -c "cd /mnt/c/path/to/project && python3 scripts/verify_phase8_max_params.py"
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple

# Phase 8 imports
try:
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig
    from src.models.phase8.hyperbolic_ssm import HyperbolicSSM, HyperbolicSSMConfig
    PHASE8_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Phase 8 modules not available: {e}")
    PHASE8_AVAILABLE = False
    sys.exit(1)


class Phase8Config:
    """Phase 8モデル設定"""
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        curvature: float = 0.01,
        dropout: float = 0.1,
        use_hyperbolic_ssm: bool = False,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.curvature = curvature
        self.dropout = dropout
        self.use_hyperbolic_ssm = use_hyperbolic_ssm


class Phase8Layer(nn.Module):
    """Phase 8 Transformer Layer"""
    def __init__(self, config: Phase8Config):
        super().__init__()
        self.config = config
        
        # Tangent-Space Linear Attention
        attn_config = LinearAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            curvature=config.curvature,
            low_curvature_threshold=0.1,
            high_curvature_threshold=1.0,
            num_features=config.d_model // config.num_heads,
            kernel_type="elu"
        )
        self.attn = TangentSpaceLinearAttention(attn_config)
        
        # Optional: Hyperbolic SSM
        if config.use_hyperbolic_ssm:
            ssm_config = HyperbolicSSMConfig(
                d_model=config.d_model,
                d_state=config.d_model // 4,
                curvature=config.curvature
            )
            self.ssm = HyperbolicSSM(ssm_config)
        else:
            self.ssm = None
        
        # FFN with Low-Rank compression
        ffn_rank = config.d_model // 8
        ffn_hidden = config.d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_rank),
            nn.Linear(ffn_rank, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_hidden, ffn_rank),
            nn.Linear(ffn_rank, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        if self.ssm is not None:
            self.ln_ssm = nn.LayerNorm(config.d_model)


class Phase8Model(nn.Module):
    """Phase 8 Language Model"""
    def __init__(self, config: Phase8Config):
        super().__init__()
        self.config = config
        
        # Low-Rank Embedding (75% compression)
        embed_rank = config.d_model // 4
        self.embed_low = nn.Embedding(config.vocab_size, embed_rank)
        self.embed_high = nn.Linear(embed_rank, config.d_model)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            Phase8Layer(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """モデルのパラメータ数を詳細にカウント"""
    param_counts = {}
    
    # Embedding
    embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n)
    param_counts['embedding'] = embed_params
    
    # Layers
    layer_params = sum(p.numel() for n, p in model.named_parameters() if 'layers' in n)
    param_counts['layers'] = layer_params
    
    # Output head
    head_params = sum(p.numel() for n, p in model.named_parameters() if 'lm_head' in n or 'ln_f' in n)
    param_counts['output_head'] = head_params
    
    # Total
    total_params = sum(p.numel() for p in model.parameters())
    param_counts['total'] = total_params
    
    return param_counts


def calculate_compression_ratio(model: nn.Module, config: Phase8Config) -> Tuple[int, int, float]:
    """圧縮率を計算（Low-Rank圧縮なしの場合と比較）"""
    
    # 実際のパラメータ数
    actual_params = sum(p.numel() for p in model.parameters())
    
    # 圧縮なしの場合のパラメータ数を計算
    # Embedding: vocab_size * d_model (圧縮なし)
    uncompressed_embed = config.vocab_size * config.d_model
    
    # Layers: 各レイヤーのパラメータ数
    # Attention: 4 * d_model * d_model (Q, K, V, O)
    # FFN: d_model * (4*d_model) + (4*d_model) * d_model (圧縮なし)
    # LayerNorm: 2 * d_model (2つのLN)
    uncompressed_layer = (
        4 * config.d_model * config.d_model +  # Attention
        config.d_model * (4 * config.d_model) + (4 * config.d_model) * config.d_model +  # FFN
        2 * config.d_model  # LayerNorm
    )
    uncompressed_layers = uncompressed_layer * config.n_layers
    
    # Output head: d_model * vocab_size
    uncompressed_head = config.d_model * config.vocab_size
    
    # Total uncompressed
    uncompressed_total = uncompressed_embed + uncompressed_layers + uncompressed_head
    
    # Compression ratio
    compression_ratio = (1 - actual_params / uncompressed_total) * 100
    
    return actual_params, uncompressed_total, compression_ratio


def load_config_from_yaml(config_path: str) -> Phase8Config:
    """YAMLファイルから設定を読み込む"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    model_cfg = config_dict.get('model', {})
    
    return Phase8Config(
        d_model=model_cfg.get('d_model', 512),
        n_layers=model_cfg.get('n_layers', 6),
        num_heads=model_cfg.get('num_heads', 8),
        vocab_size=50257,  # GPT-2 vocab size
        max_seq_len=model_cfg.get('n_seq', 512),
        curvature=model_cfg.get('curvature', 0.01),
        dropout=model_cfg.get('dropout', 0.1),
        use_hyperbolic_ssm=model_cfg.get('use_hyperbolic_ssm', False)
    )


def main():
    import sys
    
    print("=" * 80)
    print("Phase 8 Max Parameters Verification")
    print("=" * 80)
    print()
    
    # Load config (allow command line argument)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/phase8_max_push.yaml"
    
    print(f"📄 Loading config: {config_path}")
    
    try:
        config = load_config_from_yaml(config_path)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"✓ Config loaded successfully")
    print()
    
    # Display config
    print("📋 Model Configuration:")
    print(f"  d_model:           {config.d_model}")
    print(f"  n_layers:          {config.n_layers}")
    print(f"  num_heads:         {config.num_heads}")
    print(f"  vocab_size:        {config.vocab_size}")
    print(f"  max_seq_len:       {config.max_seq_len}")
    print(f"  curvature:         {config.curvature}")
    print(f"  use_hyperbolic_ssm: {config.use_hyperbolic_ssm}")
    print()
    
    # Create model
    print("🔨 Creating model...")
    try:
        model = Phase8Model(config)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    
    # Count parameters
    print("🔢 Counting parameters...")
    param_counts = count_parameters(model)
    
    print()
    print("=" * 80)
    print("📊 Parameter Count Results")
    print("=" * 80)
    print()
    
    print(f"Embedding:         {param_counts['embedding']:>15,} ({param_counts['embedding']/1e6:>8.2f}M)")
    print(f"Transformer Layers: {param_counts['layers']:>15,} ({param_counts['layers']/1e6:>8.2f}M)")
    print(f"Output Head:       {param_counts['output_head']:>15,} ({param_counts['output_head']/1e6:>8.2f}M)")
    print("-" * 80)
    print(f"Total Parameters:  {param_counts['total']:>15,} ({param_counts['total']/1e6:>8.2f}M)")
    print()
    
    # Calculate compression
    print("=" * 80)
    print("🗜️  Compression Analysis")
    print("=" * 80)
    print()
    
    actual_params, uncompressed_params, compression_ratio = calculate_compression_ratio(model, config)
    
    print(f"Without Compression: {uncompressed_params:>15,} ({uncompressed_params/1e6:>8.2f}M)")
    print(f"With Compression:    {actual_params:>15,} ({actual_params/1e6:>8.2f}M)")
    print(f"Compression Ratio:   {compression_ratio:>14.2f}%")
    print()
    
    # Memory estimation
    print("=" * 80)
    print("💾 Memory Estimation (FP16)")
    print("=" * 80)
    print()
    
    # FP16: 2 bytes per parameter
    model_memory_fp16 = actual_params * 2 / 1024**3
    print(f"Model Size (FP16):   {model_memory_fp16:>8.2f} GB")
    print()
    
    # Verification
    print("=" * 80)
    print("✅ Verification Results")
    print("=" * 80)
    print()
    
    target_params = 300_000_000  # 3億パラメータ
    target_compressed = 30_000_000  # 30M
    
    # Check if close to target
    params_diff_pct = abs(actual_params - target_params) / target_params * 100
    compressed_size_mb = actual_params / 1e6
    
    print(f"Target Parameters:     {target_params:>15,} ({target_params/1e6:>8.2f}M)")
    print(f"Actual Parameters:     {actual_params:>15,} ({actual_params/1e6:>8.2f}M)")
    print(f"Difference:            {params_diff_pct:>14.2f}%")
    print()
    
    if params_diff_pct < 10:
        print("✅ パラメータ数は目標に近い値です")
    else:
        print("⚠️  パラメータ数が目標から大きく外れています")
    
    print()
    print(f"Compressed Size:       {compressed_size_mb:>8.2f}M")
    
    if compressed_size_mb <= target_compressed * 1.1:  # 10% margin
        print("✅ 圧縮後のサイズは目標範囲内です")
    else:
        print("⚠️  圧縮後のサイズが目標を超えています")
    
    print()
    print("=" * 80)
    print("🎯 Summary")
    print("=" * 80)
    print()
    print(f"✓ Model has {actual_params/1e6:.2f}M parameters")
    print(f"✓ Compression reduces size by {compression_ratio:.2f}%")
    print(f"✓ Model size in FP16: {model_memory_fp16:.2f} GB")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
