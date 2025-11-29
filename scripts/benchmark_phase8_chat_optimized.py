#!/usr/bin/env python3
"""
Phase 8 Chat AI Optimized Benchmark
Phase 7と同じ条件でPhase 8の最適なモデル構成を探索

目標:
- 8GB VRAM環境でのチャットAI最適化
- d_model, n_layers, seq_lenの最適な組み合わせを発見
- Phase 7の1.83Bパラメータを超える構成を探索
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Phase 8モジュールのインポート
try:
    from src.models.phase8.hyperbolic_ssm import HyperbolicSSM
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention
    from src.models.phase8.block_distance import BlockWiseDistanceComputation
    from src.models.phase8.entailment_cones import EntailmentConeModule
    from src.models.phase8.sheaf_attention import SheafAttentionModule
    from src.models.phase8.quantization import LogarithmicQuantizer
    PHASE8_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 8 modules not fully available: {e}")
    PHASE8_AVAILABLE = False


@dataclass
class ModelConfig:
    """モデル構成"""
    d_model: int
    n_layers: int
    num_heads: int
    seq_len: int
    batch_size: int
    vocab_size: int = 50257
    
    # Phase 8固有
    use_hyperbolic_ssm: bool = True
    use_linear_attention: bool = True
    use_block_distance: bool = True
    use_entailment_cones: bool = False  # オプション
    use_sheaf_attention: bool = False   # オプション
    use_quantization: bool = False      # オプション
    
    # 最適化
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    def estimate_parameters(self) -> float:
        """パラメータ数を推定（百万単位）"""
        # Embedding
        embed_params = self.vocab_size * self.d_model
        
        # 各レイヤー
        layer_params = 0
        
        # Hyperbolic SSM
        if self.use_hyperbolic_ssm:
            layer_params += self.d_model * self.d_model * 3  # A, B, C matrices
        
        # Linear Attention
        if self.use_linear_attention:
            layer_params += self.d_model * self.d_model * 4  # Q, K, V, O
        
        # Block Distance (軽量)
        if self.use_block_distance:
            layer_params += self.d_model * 128  # 軽量な距離計算
        
        # FFN (standard)
        ffn_hidden = self.d_model * 4
        layer_params += self.d_model * ffn_hidden * 2
        
        # LayerNorm
        layer_params += self.d_model * 4
        
        total_params = embed_params + layer_params * self.n_layers
        
        # Output projection
        total_params += self.d_model * self.vocab_size
        
        return total_params / 1_000_000  # 百万単位


class Phase8ChatModel(nn.Module):
    """Phase 8 チャットAI最適化モデル"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Layers
        self.layers = nn.ModuleList([
            Phase8Layer(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class Phase8Layer(nn.Module):
    """Phase 8 単一レイヤー"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Hyperbolic SSM
        if config.use_hyperbolic_ssm and PHASE8_AVAILABLE:
            self.ssm = HyperbolicSSM(
                d_model=config.d_model,
                curvature=1.0
            )
        else:
            self.ssm = None
        
        # Linear Attention
        if config.use_linear_attention and PHASE8_AVAILABLE:
            self.attn = TangentSpaceLinearAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                curvature=1.0
            )
        else:
            # Fallback to standard attention
            self.attn = nn.MultiheadAttention(
                config.d_model,
                config.num_heads,
                batch_first=True
            )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM
        if self.ssm is not None:
            x = x + self.ssm(self.ln1(x))
        
        # Attention
        residual = x
        x = self.ln1(x)
        if isinstance(self.attn, nn.MultiheadAttention):
            x, _ = self.attn(x, x, x, need_weights=False)
        else:
            x = self.attn(x)
        x = residual + x
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        
        return x


def benchmark_config(config: ModelConfig, device: str = 'cuda') -> Dict:
    """単一構成のベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Testing: d_model={config.d_model}, layers={config.n_layers}, seq={config.seq_len}")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return None
    
    try:
        # モデル作成
        model = Phase8ChatModel(config).to(device)
        
        if config.mixed_precision:
            model = model.half()
        
        # ダミー入力
        input_ids = torch.randint(
            0, config.vocab_size,
            (config.batch_size, config.seq_len),
            device=device
        )
        
        # メモリ測定
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            logits = model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # スループット測定
        torch.cuda.synchronize()
        start = time.time()
        
        n_iters = 10
        for _ in range(n_iters):
            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                _ = model(input_ids)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tokens_per_sec = (config.batch_size * config.seq_len * n_iters) / elapsed
        
        # パラメータ数
        params_m = config.estimate_parameters()
        
        result = {
            'config': asdict(config),
            'parameters_millions': round(params_m, 2),
            'parameters_billions': round(params_m / 1000, 3),
            'peak_memory_gb': round(peak_memory, 3),
            'tokens_per_sec': int(tokens_per_sec),
            'status': 'SUCCESS'
        }
        
        print(f"✓ Parameters: {result['parameters_billions']:.2f}B")
        print(f"✓ VRAM: {result['peak_memory_gb']:.2f} GB")
        print(f"✓ Throughput: {result['tokens_per_sec']:,} tok/s")
        
        # クリーンアップ
        del model, logits, input_ids
        torch.cuda.empty_cache()
        
        return result
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"✗ OOM: {str(e)[:100]}")
            torch.cuda.empty_cache()
            return {
                'config': asdict(config),
                'status': 'OOM',
                'error': 'Out of memory'
            }
        else:
            print(f"✗ Error: {str(e)[:100]}")
            return {
                'config': asdict(config),
                'status': 'ERROR',
                'error': str(e)[:200]
            }


def main():
    """メインベンチマーク"""
    print("="*60)
    print("Phase 8 Chat AI Optimized Benchmark")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {total_memory:.2f} GB")
    
    # テスト構成
    # Phase 7の1.83Bを基準に、Phase 8の最適化を活かした構成を探索
    configs = []
    
    # 1. 小規模テスト（動作確認）
    configs.append(ModelConfig(
        d_model=512, n_layers=8, num_heads=8,
        seq_len=512, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 2. 中規模（チャットAI推奨）
    configs.append(ModelConfig(
        d_model=1024, n_layers=16, num_heads=16,
        seq_len=512, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 3. 大規模（Phase 7相当を目指す）
    configs.append(ModelConfig(
        d_model=2048, n_layers=24, num_heads=16,
        seq_len=512, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 4. 長文対応
    configs.append(ModelConfig(
        d_model=1024, n_layers=16, num_heads=16,
        seq_len=1024, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 5. 超長文対応（Phase 8の強み）
    configs.append(ModelConfig(
        d_model=1024, n_layers=16, num_heads=16,
        seq_len=2048, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 6. バッチ訓練
    configs.append(ModelConfig(
        d_model=1024, n_layers=16, num_heads=16,
        seq_len=512, batch_size=2,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # 7. 深層モデル
    configs.append(ModelConfig(
        d_model=1536, n_layers=32, num_heads=12,
        seq_len=512, batch_size=1,
        use_hyperbolic_ssm=True,
        use_linear_attention=True,
        use_block_distance=True,
    ))
    
    # ベンチマーク実行
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        result = benchmark_config(config, device)
        if result:
            results.append(result)
    
    # 結果保存
    output_dir = Path('results/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'PHASE8_CHAT_OPTIMIZED_BENCHMARK.json'
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_name,
        'total_memory_gb': round(total_memory, 2),
        'phase8_available': PHASE8_AVAILABLE,
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    # 成功した構成を表示
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        # パラメータ数でソート
        successful.sort(key=lambda x: x['parameters_millions'], reverse=True)
        
        print("\n✓ Successful Configurations:")
        print(f"{'Params (B)':<12} {'VRAM (GB)':<12} {'Throughput':<15} {'Config'}")
        print("-" * 70)
        
        for r in successful:
            cfg = r['config']
            print(f"{r['parameters_billions']:<12.2f} "
                  f"{r['peak_memory_gb']:<12.2f} "
                  f"{r['tokens_per_sec']:>10,} tok/s  "
                  f"d={cfg['d_model']}, L={cfg['n_layers']}, seq={cfg['seq_len']}")
        
        # 最大構成
        max_config = successful[0]
        print(f"\n🏆 Maximum Configuration:")
        print(f"   Parameters: {max_config['parameters_billions']:.2f}B")
        print(f"   VRAM: {max_config['peak_memory_gb']:.2f} GB")
        print(f"   Throughput: {max_config['tokens_per_sec']:,} tok/s")
        
        # Phase 7との比較
        phase7_params = 1.83
        phase7_vram = 6.89
        
        print(f"\n📊 Phase 7 Comparison:")
        print(f"   Parameters: {phase7_params:.2f}B → {max_config['parameters_billions']:.2f}B "
              f"({(max_config['parameters_billions']/phase7_params - 1)*100:+.1f}%)")
        print(f"   VRAM: {phase7_vram:.2f}GB → {max_config['peak_memory_gb']:.2f}GB "
              f"({(max_config['peak_memory_gb']/phase7_vram - 1)*100:+.1f}%)")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
