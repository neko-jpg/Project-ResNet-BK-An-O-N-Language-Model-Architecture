"""
Phase 8 Quick Triton Benchmark Script

WSL環境でTritonカーネルの動作を確認するための簡易ベンチマーク
"""
import torch
import torch.nn as nn
import time
import json
import gc
from datetime import datetime
import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import triton
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None

from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config


def benchmark_quick(model, input_ids, num_iterations=10):
    """クイックベンチマーク"""
    batch_size, seq_len = input_ids.shape
    
    # ウォームアップ
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, return_diagnostics=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # 測定
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(input_ids, return_diagnostics=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    mean_time = sum(times) / len(times)
    tokens_per_sec = (batch_size * seq_len) / mean_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'mean_time_sec': mean_time,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_mb': peak_memory_mb,
    }


def main():
    print("="*60)
    print("Phase 8 Quick Triton Benchmark")
    print("="*60)
    
    print(f"\nTriton Available: {TRITON_AVAILABLE}")
    if TRITON_AVAILABLE:
        print(f"Triton Version: {TRITON_VERSION}")
    
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    device = 'cuda'
    results = {
        'timestamp': datetime.now().isoformat(),
        'triton_available': TRITON_AVAILABLE,
        'triton_version': TRITON_VERSION,
        'gpu_name': torch.cuda.get_device_name(0),
        'benchmarks': []
    }
    
    # テスト設定
    test_configs = [
        # 小型モデル - 複数のシーケンス長
        ('Phase8_Small_seq256', 512, 8, 256, [1, 2, 4, 8]),
        ('Phase8_Small_seq512', 512, 8, 512, [1, 2, 4]),
        ('Phase8_Small_seq1024', 512, 8, 1024, [1, 2]),
        # 中型モデル
        ('Phase8_Medium_seq512', 1024, 16, 512, [1, 2]),
    ]
    
    for name, d_model, n_layers, n_seq, batch_sizes in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"  d_model={d_model}, n_layers={n_layers}, n_seq={n_seq}")
        print(f"{'='*60}")
        
        try:
            config = Phase8Config(
                vocab_size=50257,
                d_model=d_model,
                n_layers=n_layers,
                n_seq=n_seq,
                num_heads=8 if d_model == 512 else 16,
                htt_rank=16 if d_model == 512 else 32,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
                use_triton_kernel=True,
                triton_kernel_version='fast',
            )
            
            model = Phase8IntegratedModel(config).to(device).eval()
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params/1e6:.2f}M")
            
            for batch_size in batch_sizes:
                print(f"\n  batch_size={batch_size}, seq_len={n_seq}")
                
                try:
                    input_ids = torch.randint(0, config.vocab_size, (batch_size, n_seq), device=device)
                    result = benchmark_quick(model, input_ids, num_iterations=10)
                    
                    print(f"    Tokens/sec: {result['tokens_per_sec']:.0f}")
                    print(f"    Peak memory: {result['peak_memory_mb']:.2f} MB")
                    print(f"    Time: {result['mean_time_sec']*1000:.2f} ms")
                    
                    results['benchmarks'].append({
                        'config_name': name,
                        'd_model': d_model,
                        'n_layers': n_layers,
                        'n_seq': n_seq,
                        'total_params_millions': total_params / 1e6,
                        **result
                    })
                    
                except RuntimeError as e:
                    print(f"    ERROR: {str(e)[:100]}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # 結果を保存
    output_file = 'results/benchmarks/phase8_triton_quick_benchmark.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # サマリー
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for bench in results['benchmarks']:
        print(f"\n{bench['config_name']} (batch={bench['batch_size']})")
        print(f"  Params: {bench['total_params_millions']:.2f}M")
        print(f"  Throughput: {bench['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Memory: {bench['peak_memory_mb']:.2f} MB")


if __name__ == '__main__':
    main()
