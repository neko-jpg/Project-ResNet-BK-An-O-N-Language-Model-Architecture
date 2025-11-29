#!/usr/bin/env python3
"""
Phase 7 vs Phase 8 公平比較ベンチマーク
Phase 7と全く同じ条件でPhase 8を測定
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import json
import time
from dataclasses import dataclass

# Phase 7の設定を読み込み
phase7_config = {
    "d_model": 4096,
    "n_layers": 32,
    "batch_size": 1,
    "seq_len": 512,
    "vocab_size": 50257,
    "gradient_checkpointing": True,
    "mixed_precision": True,
    "low_rank_embedding": True,
    "low_rank_ffn": True,
}


@dataclass
class TestConfig:
    name: str
    d_model: int
    n_layers: int
    num_heads: int
    use_phase8: bool = False


def estimate_params(d_model, n_layers, vocab_size=50257):
    """パラメータ数を推定"""
    # Embedding (低ランク圧縮: 75%)
    embed_rank = d_model // 4
    embed_params = vocab_size * embed_rank + embed_rank * d_model
    
    # 各レイヤー
    layer_params = 0
    
    # Attention (Q, K, V, O)
    layer_params += d_model * d_model * 4
    
    # FFN (低ランク圧縮: 87.5%)
    ffn_rank = d_model // 8
    ffn_hidden = d_model * 4
    layer_params += d_model * ffn_rank + ffn_rank * ffn_hidden
    layer_params += ffn_hidden * ffn_rank + ffn_rank * d_model
    
    # LayerNorm
    layer_params += d_model * 4
    
    total_params = embed_params + layer_params * n_layers
    total_params += d_model * vocab_size  # Output projection
    
    return total_params / 1_000_000


class SimpleLayer(nn.Module):
    """Phase 7スタイルの標準レイヤー"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 低ランクFFN
        ffn_rank = d_model // 8
        ffn_hidden = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_rank),
            nn.Linear(ffn_rank, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, ffn_rank),
            nn.Linear(ffn_rank, d_model)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        x = x + self.ffn(self.ln2(x))
        return x


class Phase7Model(nn.Module):
    """Phase 7スタイルモデル"""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
        # 低ランクEmbedding
        embed_rank = config.d_model // 4
        self.embed_low = nn.Embedding(50257, embed_rank)
        self.embed_high = nn.Linear(embed_rank, config.d_model)
        
        self.layers = nn.ModuleList([
            SimpleLayer(config.d_model, config.num_heads)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, 50257, bias=False)
        
    def forward(self, input_ids):
        x = self.embed_high(self.embed_low(input_ids))
        
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        return self.lm_head(x)


try:
    from src.models.phase8.hyperbolic_ssm import HyperbolicSSM, HyperbolicSSMConfig
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig
    PHASE8_AVAILABLE = True
except:
    PHASE8_AVAILABLE = False


class Phase8Layer(nn.Module):
    """Phase 8レイヤー（Phase 7と同じ最適化を適用）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        if PHASE8_AVAILABLE:
            # Linear Attention ONLY (Phase 7のMultiheadAttentionと置き換え)
            # 低曲率モードで線形計算のみを使用
            attn_config = LinearAttentionConfig(
                d_model=d_model,
                num_heads=num_heads,
                curvature=0.01,  # 低曲率で線形モードを強制
                low_curvature_threshold=0.1,
                high_curvature_threshold=1.0,
                num_features=d_model // num_heads,  # 特徴数を適切に設定
                kernel_type="elu"
            )
            self.attn = TangentSpaceLinearAttention(attn_config)
        else:
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 低ランクFFN (Phase 7と同じ)
        ffn_rank = d_model // 8
        ffn_hidden = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_rank),
            nn.Linear(ffn_rank, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, ffn_rank),
            nn.Linear(ffn_rank, d_model)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        if isinstance(self.attn, nn.MultiheadAttention):
            x, _ = self.attn(x, x, x, need_weights=False)
        else:
            attn_out = self.attn(x)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = attn_out
        x = residual + x
        x = x + self.ffn(self.ln2(x))
        return x


class Phase8Model(nn.Module):
    """Phase 8モデル（Phase 7と同じ最適化を適用）"""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
        # 低ランクEmbedding (Phase 7と同じ)
        embed_rank = config.d_model // 4
        self.embed_low = nn.Embedding(50257, embed_rank)
        self.embed_high = nn.Linear(embed_rank, config.d_model)
        
        self.layers = nn.ModuleList([
            Phase8Layer(config.d_model, config.num_heads)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, 50257, bias=False)
        
    def forward(self, input_ids):
        x = self.embed_high(self.embed_low(input_ids))
        
        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        return self.lm_head(x)


def benchmark_config(config: TestConfig):
    """単一構成のベンチマーク"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.device('cuda')
    print(f"\n{'='*60}")
    print(f"{config.name}")
    print(f"d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"{'='*60}")
    
    try:
        # メモリクリア
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # モデル作成
        if config.use_phase8:
            model = Phase8Model(config)
        else:
            model = Phase7Model(config)
        
        # FP16に変換してGPUに転送
        model = model.half().to(device)
        model.eval()  # 評価モード
        
        # モデルパラメータのメモリ
        model_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Model Memory: {model_memory:.2f} GB")
        
        # 入力データ
        input_ids = torch.randint(0, 50257, (1, 512), device=device)
        
        # Forward pass with gradient checkpointing disabled for inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                _ = model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        params_m = estimate_params(config.d_model, config.n_layers)
        
        result = {
            'name': config.name,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'parameters_millions': round(params_m, 2),
            'parameters_billions': round(params_m / 1000, 3),
            'model_memory_gb': round(model_memory, 3),
            'peak_memory_gb': round(peak_memory, 3),
            'activation_memory_gb': round(peak_memory - model_memory, 3),
            'status': 'SUCCESS'
        }
        
        print(f"Parameters: {result['parameters_billions']:.2f}B")
        print(f"Model VRAM: {result['model_memory_gb']:.2f} GB")
        print(f"Peak VRAM: {result['peak_memory_gb']:.2f} GB")
        print(f"Activation VRAM: {result['activation_memory_gb']:.2f} GB")
        
        del model, input_ids
        torch.cuda.empty_cache()
        return result
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"OOM")
            torch.cuda.empty_cache()
            return {'name': config.name, 'status': 'OOM'}
        else:
            print(f"ERROR: {str(e)[:100]}")
            torch.cuda.empty_cache()
            return {'name': config.name, 'status': 'ERROR'}


def main():
    print("="*60)
    print("Phase 7 vs Phase 8 Fair Comparison")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {total_memory:.2f} GB")
    print(f"Phase 8 Available: {PHASE8_AVAILABLE}")
    
    # テスト構成（Phase 7の実際の構成）
    configs = [
        # Phase 7 Maximum (1.83B)
        TestConfig("Phase 7 - Maximum (1.83B)", 4096, 32, 32, use_phase8=False),
        TestConfig("Phase 8 - Maximum (1.83B)", 4096, 32, 32, use_phase8=True),
        
        # Phase 7 Large
        TestConfig("Phase 7 - Large (1.54B)", 3072, 48, 24, use_phase8=False),
        TestConfig("Phase 8 - Large (1.54B)", 3072, 48, 24, use_phase8=True),
        
        # Phase 7 Deep
        TestConfig("Phase 7 - Deep (0.92B)", 2048, 64, 16, use_phase8=False),
        TestConfig("Phase 8 - Deep (0.92B)", 2048, 64, 16, use_phase8=True),
        
        # Phase 7 Standard
        TestConfig("Phase 7 - Standard (0.70B)", 2048, 48, 16, use_phase8=False),
        TestConfig("Phase 8 - Standard (0.70B)", 2048, 48, 16, use_phase8=True),
    ]
    
    results = []
    for config in configs:
        result = benchmark_config(config)
        if result:
            results.append(result)
    
    # 結果保存
    output_file = Path('results/benchmarks/PHASE7_VS_PHASE8_COMPARISON.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_name,
        'total_memory_gb': round(total_memory, 2),
        'phase8_available': PHASE8_AVAILABLE,
        'optimization_settings': {
            'gradient_checkpointing': True,
            'mixed_precision': 'FP16',
            'low_rank_embedding': '75% compression (d_model/4)',
            'low_rank_ffn': '87.5% compression (d_model/8)',
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # サマリー表示
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"\n{'Model':<35} {'Params (B)':<12} {'Model (GB)':<12} {'Peak (GB)':<12} {'Act (GB)':<12}")
        print("-" * 85)
        
        for r in successful:
            print(f"{r['name']:<35} {r['parameters_billions']:<12.2f} {r['model_memory_gb']:<12.2f} {r['peak_memory_gb']:<12.2f} {r['activation_memory_gb']:<12.2f}")
        
        # Phase 7 vs Phase 8比較
        phase7_results = [r for r in successful if 'Phase 7' in r['name']]
        phase8_results = [r for r in successful if 'Phase 8' in r['name']]
        
        if phase7_results and phase8_results:
            print(f"\n{'='*60}")
            print("Phase 7 vs Phase 8 Direct Comparison")
            print(f"{'='*60}")
            
            for p7, p8 in zip(phase7_results, phase8_results):
                if p7['parameters_billions'] == p8['parameters_billions']:
                    model_diff = p8['model_memory_gb'] - p7['model_memory_gb']
                    peak_diff = p8['peak_memory_gb'] - p7['peak_memory_gb']
                    model_pct = (model_diff / p7['model_memory_gb']) * 100 if p7['model_memory_gb'] > 0 else 0
                    peak_pct = (peak_diff / p7['peak_memory_gb']) * 100 if p7['peak_memory_gb'] > 0 else 0
                    
                    print(f"\n{p7['parameters_billions']:.2f}B Model:")
                    print(f"  Phase 7 Model: {p7['model_memory_gb']:.2f} GB | Peak: {p7['peak_memory_gb']:.2f} GB")
                    print(f"  Phase 8 Model: {p8['model_memory_gb']:.2f} GB | Peak: {p8['peak_memory_gb']:.2f} GB")
                    print(f"  Model Diff: {model_diff:+.2f} GB ({model_pct:+.1f}%)")
                    print(f"  Peak Diff: {peak_diff:+.2f} GB ({peak_pct:+.1f}%)")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
