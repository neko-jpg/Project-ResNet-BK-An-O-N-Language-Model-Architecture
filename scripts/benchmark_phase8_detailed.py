#!/usr/bin/env python3
"""Phase 8 Detailed Benchmark Script"""

import torch
import time
import json
from datetime import datetime

# GPU情報
gpu_info = {}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_info = {
        'name': props.name,
        'total_memory_gb': props.total_memory / (1024**3),
        'compute_capability': f'{props.major}.{props.minor}',
    }
    print(f'GPU: {gpu_info["name"]}, VRAM: {gpu_info["total_memory_gb"]:.1f} GB')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Phase 8モジュールのスループットテスト
results = {
    'timestamp': datetime.now().isoformat(),
    'gpu_info': gpu_info,
    'benchmarks': {},
    'targets': {
        'throughput_improvement': '2x over Phase 7',
        'memory_reduction': '50-80%',
        'max_memory_8192': '<3GB',
        'flops_utilization': '70%+',
        'linear_scaling': 'O(N)',
    }
}

# TangentSpaceLinearAttention テスト
print('\nTesting TangentSpaceLinearAttention...')
from src.models.phase8 import TangentSpaceLinearAttention, LinearAttentionConfig
config = LinearAttentionConfig(d_model=256, num_heads=4)
model = TangentSpaceLinearAttention(config).to(device)
model.eval()

for seq_len in [512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # 測定
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_tokens = 2 * seq_len * 20
    tokens_per_sec = total_tokens / (end - start)
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f'  seq={seq_len}: {tokens_per_sec:.0f} tok/s, {memory_mb:.1f} MB')
    results['benchmarks'][f'linear_attention_seq{seq_len}'] = {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb,
        'seq_len': seq_len
    }

del model
torch.cuda.empty_cache()

# HyperbolicSSM テスト
print('\nTesting HyperbolicSSM...')
from src.models.phase8 import HyperbolicSSM, HyperbolicSSMConfig
config = HyperbolicSSMConfig(d_model=256)
model = HyperbolicSSM(config).to(device)
model.eval()

for seq_len in [512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # 測定
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_tokens = 2 * seq_len * 20
    tokens_per_sec = total_tokens / (end - start)
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f'  seq={seq_len}: {tokens_per_sec:.0f} tok/s, {memory_mb:.1f} MB')
    results['benchmarks'][f'hyperbolic_ssm_seq{seq_len}'] = {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb,
        'seq_len': seq_len
    }

del model
torch.cuda.empty_cache()

# BlockWiseDistanceComputation テスト
print('\nTesting BlockWiseDistanceComputation...')
from src.models.phase8 import BlockWiseDistanceComputation, BlockDistanceConfig
config = BlockDistanceConfig(d_model=256, num_heads=4)
model = BlockWiseDistanceComputation(config).to(device)
model.eval()

for seq_len in [512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # 測定
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_tokens = 2 * seq_len * 20
    tokens_per_sec = total_tokens / (end - start)
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f'  seq={seq_len}: {tokens_per_sec:.0f} tok/s, {memory_mb:.1f} MB')
    results['benchmarks'][f'block_distance_seq{seq_len}'] = {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb,
        'seq_len': seq_len
    }

del model
torch.cuda.empty_cache()

# SheafAttentionModule テスト
print('\nTesting SheafAttentionModule...')
from src.models.phase8 import SheafAttentionModule, SheafAttentionConfig
config = SheafAttentionConfig(d_model=256, num_heads=4)
model = SheafAttentionModule(config).to(device)
model.eval()

for seq_len in [512, 1024]:
    x = torch.randn(2, seq_len, 256, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # 測定
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_tokens = 2 * seq_len * 20
    tokens_per_sec = total_tokens / (end - start)
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f'  seq={seq_len}: {tokens_per_sec:.0f} tok/s, {memory_mb:.1f} MB')
    results['benchmarks'][f'sheaf_attention_seq{seq_len}'] = {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb,
        'seq_len': seq_len
    }

del model
torch.cuda.empty_cache()

# LogarithmicQuantizer テスト
print('\nTesting LogarithmicQuantizer...')
from src.models.phase8 import LogarithmicQuantizer, QuantizationConfig
config = QuantizationConfig(bits=8)
model = LogarithmicQuantizer(config).to(device)
model.eval()

for seq_len in [512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
    
    # 測定
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_tokens = 2 * seq_len * 50
    tokens_per_sec = total_tokens / (end - start)
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f'  seq={seq_len}: {tokens_per_sec:.0f} tok/s, {memory_mb:.1f} MB')
    results['benchmarks'][f'quantizer_seq{seq_len}'] = {
        'tokens_per_sec': tokens_per_sec,
        'memory_mb': memory_mb,
        'seq_len': seq_len
    }

del model
torch.cuda.empty_cache()

# メモリスケーリングテスト（O(N)確認）- HyperbolicSSMを使用（真のO(N)）
print('\nMemory Scaling Test (O(N) verification) - HyperbolicSSM...')
from src.models.phase8 import HyperbolicSSM, HyperbolicSSMConfig
config = HyperbolicSSMConfig(d_model=256)
model = HyperbolicSSM(config).to(device)
model.eval()

memory_scaling = []
for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        x = torch.randn(2, seq_len, 256, device=device)
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        memory_scaling.append({'seq_len': seq_len, 'memory_mb': memory_mb})
        print(f'  seq={seq_len}: {memory_mb:.1f} MB')
    except Exception as e:
        print(f'  seq={seq_len}: OOM - {e}')
        break

results['memory_scaling_hyperbolic_ssm'] = memory_scaling

# O(N)スケーリング検証
if len(memory_scaling) >= 2:
    # seq_len が2倍になったときのメモリ増加率を計算
    ratios = []
    for i in range(1, len(memory_scaling)):
        seq_ratio = memory_scaling[i]['seq_len'] / memory_scaling[i-1]['seq_len']
        mem_ratio = memory_scaling[i]['memory_mb'] / memory_scaling[i-1]['memory_mb']
        ratios.append(mem_ratio / seq_ratio)
    
    avg_ratio = sum(ratios) / len(ratios)
    is_linear = avg_ratio < 1.5  # O(N)なら比率は約1
    results['is_linear_scaling_ssm'] = is_linear
    results['memory_scaling_ratio_ssm'] = avg_ratio
    print(f'\nMemory scaling ratio (SSM): {avg_ratio:.2f} (target: ~1.0 for O(N))')
    print(f'Linear scaling: {"YES" if is_linear else "NO"}')

# 長コンテキストテスト - HyperbolicSSM
print('\nLong Context Test (seq=8192) - HyperbolicSSM...')
try:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(1, 8192, 256, device=device)
    with torch.no_grad():
        _ = model(x)
        torch.cuda.synchronize()
    
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    results['long_context_8192_ssm'] = {
        'memory_mb': memory_mb,
        'target_met': memory_mb < 3000,  # <3GB target
    }
    print(f'  seq=8192: {memory_mb:.1f} MB (target: <3000 MB)')
    print(f'  Target met: {"YES" if memory_mb < 3000 else "NO"}')
except Exception as e:
    results['long_context_8192_ssm'] = {'error': str(e)}
    print(f'  Error: {e}')

del model
torch.cuda.empty_cache()

# BlockWiseDistanceComputation メモリスケーリングテスト
print('\nMemory Scaling Test (O(N) verification) - BlockWiseDistance...')
from src.models.phase8 import BlockWiseDistanceComputation, BlockDistanceConfig
config = BlockDistanceConfig(d_model=256, num_heads=4)
model = BlockWiseDistanceComputation(config).to(device)
model.eval()

memory_scaling_block = []
for seq_len in [256, 512, 1024, 2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        x = torch.randn(2, seq_len, 256, device=device)
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        memory_scaling_block.append({'seq_len': seq_len, 'memory_mb': memory_mb})
        print(f'  seq={seq_len}: {memory_mb:.1f} MB')
    except Exception as e:
        print(f'  seq={seq_len}: OOM - {e}')
        break

results['memory_scaling_block_distance'] = memory_scaling_block

# O(N)スケーリング検証
if len(memory_scaling_block) >= 2:
    ratios = []
    for i in range(1, len(memory_scaling_block)):
        seq_ratio = memory_scaling_block[i]['seq_len'] / memory_scaling_block[i-1]['seq_len']
        mem_ratio = memory_scaling_block[i]['memory_mb'] / memory_scaling_block[i-1]['memory_mb']
        ratios.append(mem_ratio / seq_ratio)
    
    avg_ratio = sum(ratios) / len(ratios)
    is_linear = avg_ratio < 1.5
    results['is_linear_scaling_block'] = is_linear
    results['memory_scaling_ratio_block'] = avg_ratio
    print(f'\nMemory scaling ratio (Block): {avg_ratio:.2f} (target: ~1.0 for O(N))')
    print(f'Linear scaling: {"YES" if is_linear else "NO"}')

del model
torch.cuda.empty_cache()

# サマリー
print('\n' + '='*60)
print('BENCHMARK SUMMARY')
print('='*60)

# 目標達成状況
targets_met = {
    'linear_scaling_ssm': results.get('is_linear_scaling_ssm', False),
    'linear_scaling_block': results.get('is_linear_scaling_block', False),
    'long_context_8192': results.get('long_context_8192_ssm', {}).get('target_met', False),
}

print(f"O(N) Memory Scaling (SSM): {'✓ ACHIEVED' if targets_met['linear_scaling_ssm'] else '✗ NOT MET'}")
print(f"O(N) Memory Scaling (Block): {'✓ ACHIEVED' if targets_met['linear_scaling_block'] else '✗ NOT MET'}")
print(f"Long Context (8192): {'✓ ACHIEVED' if targets_met['long_context_8192'] else '✗ NOT MET'}")

results['targets_met'] = targets_met
results['overall_status'] = 'PASS' if all(targets_met.values()) else 'PARTIAL'

print(f"\nOverall Status: {results['overall_status']}")

# 結果保存
with open('results/benchmarks/phase8_benchmark_detailed.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to results/benchmarks/phase8_benchmark_detailed.json')
