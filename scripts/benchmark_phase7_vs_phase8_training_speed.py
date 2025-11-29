#!/usr/bin/env python3
"""
Phase 7 vs Phase 8 訓練速度比較ベンチマーク
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
import numpy as np


@dataclass
class TestConfig:
    name: str
    d_model: int
    n_layers: int
    num_heads: int
    use_phase8: bool = False


class SimpleLayer(nn.Module):
    """Phase 7スタイルの標準レイヤー"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
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
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig
    PHASE8_AVAILABLE = True
except:
    PHASE8_AVAILABLE = False


class Phase8Layer(nn.Module):
    """Phase 8レイヤー"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        if PHASE8_AVAILABLE:
            attn_config = LinearAttentionConfig(
                d_model=d_model,
                num_heads=num_heads,
                curvature=0.01,
                low_curvature_threshold=0.1,
                high_curvature_threshold=1.0,
                num_features=d_model // num_heads,
                kernel_type="elu"
            )
            self.attn = TangentSpaceLinearAttention(attn_config)
        else:
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
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
    """Phase 8モデル"""
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
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


def benchmark_training_speed(config: TestConfig, batch_size=2, seq_len=512, num_steps=20):
    """訓練速度のベンチマーク"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.device('cuda')
    print(f"\n{'='*60}")
    print(f"{config.name}")
    print(f"d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"{'='*60}")
    
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # モデル作成
        if config.use_phase8:
            model = Phase8Model(config)
        else:
            model = Phase7Model(config)
        
        model = model.half().to(device)
        model.train()
        
        # オプティマイザ
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # ウォームアップ
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        
        for _ in range(3):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 50257),
                    targets.reshape(-1)
                )
            loss.backward()
            optimizer.step()
        
        # 測定
        torch.cuda.synchronize()
        times = []
        
        for step in range(num_steps):
            start_time = time.time()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, 50257),
                    targets.reshape(-1)
                )
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            times.append(step_time)
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{num_steps}: {step_time:.3f}s")
        
        # 統計
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_sec = (batch_size * seq_len) / mean_time
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        result = {
            'name': config.name,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mean_step_time_sec': round(float(mean_time), 4),
            'std_step_time_sec': round(float(std_time), 4),
            'tokens_per_sec': round(float(tokens_per_sec), 2),
            'peak_memory_gb': round(peak_memory, 3),
            'status': 'SUCCESS'
        }
        
        print(f"\n結果:")
        print(f"  平均ステップ時間: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"  スループット: {tokens_per_sec:.2f} tokens/sec")
        print(f"  ピークメモリ: {peak_memory:.2f} GB")
        
        del model, optimizer, input_ids, targets, logits, loss
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
            return {'name': config.name, 'status': 'ERROR', 'error': str(e)[:200]}


def main():
    print("="*60)
    print("Phase 7 vs Phase 8 Training Speed Comparison")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {total_memory:.2f} GB")
    print(f"Phase 8 Available: {PHASE8_AVAILABLE}")
    
    # テスト構成（小さめのサイズで訓練速度を測定）
    configs = [
        # Small (訓練速度測定用)
        TestConfig("Phase 7 - Small", 512, 6, 8, use_phase8=False),
        TestConfig("Phase 8 - Small", 512, 6, 8, use_phase8=True),
        
        # Medium
        TestConfig("Phase 7 - Medium", 1024, 12, 16, use_phase8=False),
        TestConfig("Phase 8 - Medium", 1024, 12, 16, use_phase8=True),
        
        # Large
        TestConfig("Phase 7 - Large", 2048, 24, 16, use_phase8=False),
        TestConfig("Phase 8 - Large", 2048, 24, 16, use_phase8=True),
    ]
    
    results = []
    for config in configs:
        result = benchmark_training_speed(config, batch_size=2, seq_len=512, num_steps=20)
        if result:
            results.append(result)
    
    # 結果保存
    output_file = Path('results/benchmarks/PHASE7_VS_PHASE8_TRAINING_SPEED.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_name,
        'total_memory_gb': round(total_memory, 2),
        'phase8_available': PHASE8_AVAILABLE,
        'test_settings': {
            'batch_size': 2,
            'seq_len': 512,
            'num_steps': 20,
            'mixed_precision': 'FP16',
            'gradient_checkpointing': True,
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 比較サマリー
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        print(f"\n{'='*60}")
        print("Training Speed Comparison Summary")
        print(f"{'='*60}")
        print(f"\n{'Model':<25} {'Step Time (s)':<18} {'Tokens/sec':<15} {'Memory (GB)':<12}")
        print("-" * 70)
        
        for r in successful:
            print(f"{r['name']:<25} {r['mean_step_time_sec']:<18.4f} {r['tokens_per_sec']:<15.2f} {r['peak_memory_gb']:<12.2f}")
        
        # Phase 7 vs Phase 8 直接比較
        phase7_results = [r for r in successful if 'Phase 7' in r['name']]
        phase8_results = [r for r in successful if 'Phase 8' in r['name']]
        
        if phase7_results and phase8_results:
            print(f"\n{'='*60}")
            print("Phase 7 vs Phase 8 Speed Difference")
            print(f"{'='*60}")
            
            for p7, p8 in zip(phase7_results, phase8_results):
                if p7['d_model'] == p8['d_model']:
                    time_diff = p8['mean_step_time_sec'] - p7['mean_step_time_sec']
                    time_pct = (time_diff / p7['mean_step_time_sec']) * 100
                    
                    throughput_diff = p8['tokens_per_sec'] - p7['tokens_per_sec']
                    throughput_pct = (throughput_diff / p7['tokens_per_sec']) * 100
                    
                    print(f"\n{p7['name']} vs {p8['name']}:")
                    print(f"  Step Time: {p7['mean_step_time_sec']:.4f}s vs {p8['mean_step_time_sec']:.4f}s ({time_pct:+.1f}%)")
                    print(f"  Throughput: {p7['tokens_per_sec']:.2f} vs {p8['tokens_per_sec']:.2f} tokens/sec ({throughput_pct:+.1f}%)")
                    
                    if time_pct < 0:
                        print(f"  → Phase 8 is {abs(time_pct):.1f}% FASTER! 🚀")
                    elif time_pct > 0:
                        print(f"  → Phase 7 is {time_pct:.1f}% faster")
                    else:
                        print(f"  → Same speed")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
