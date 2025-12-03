"""
Phase 8 Comprehensive Benchmark Script

厳密なベンチマーク測定:
1. 統合モデルの実測スループット
2. メモリ使用量の詳細測定
3. 複数のバッチサイズとシーケンス長
4. ウォームアップとクールダウン
5. 統計的に有意な測定（複数回実行）
6. Tritonカーネルの使用を確認
"""
import torch
import torch.nn as nn
import time
import json
import gc
from datetime import datetime
from typing import Dict, List, Any
import sys
import os
import warnings

# 警告を抑制
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Tritonの確認
try:
    import triton
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = None
    print("WARNING: Triton not available. Install with: pip install triton")

from src.models.phase8.integrated_model import Phase8IntegratedModel, create_phase8_model
from src.models.phase8.config import Phase8Config


def get_gpu_memory():
    """GPU メモリ使用量を取得（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_gpu_memory():
    """ピークGPUメモリ使用量を取得（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_peak_memory():
    """ピークメモリをリセット"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def warmup_model(model, input_ids, num_iterations=5):
    """モデルのウォームアップ"""
    print(f"  Warming up ({num_iterations} iterations)...")
    model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids, return_diagnostics=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # メモリをクリア
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_throughput(
    model,
    input_ids,
    num_iterations=20,
    warmup_iterations=5
):
    """
    スループットを厳密に測定
    
    Returns:
        Dict with throughput metrics
    """
    batch_size, seq_len = input_ids.shape
    
    # ウォームアップ
    warmup_model(model, input_ids, warmup_iterations)
    
    # メモリをリセット
    reset_peak_memory()
    
    # 測定開始
    model.eval()
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_ids, return_diagnostics=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
    
    # 統計計算
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    max_time = times.max().item()
    
    # トークン/秒を計算
    tokens = batch_size * seq_len
    tokens_per_sec_mean = tokens / mean_time
    tokens_per_sec_min = tokens / max_time  # 最悪ケース
    tokens_per_sec_max = tokens / min_time  # 最良ケース
    
    # メモリ測定
    peak_memory_mb = get_peak_gpu_memory()
    current_memory_mb = get_gpu_memory()
    
    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'total_tokens': tokens,
        'num_iterations': num_iterations,
        'mean_time_sec': mean_time,
        'std_time_sec': std_time,
        'min_time_sec': min_time,
        'max_time_sec': max_time,
        'tokens_per_sec_mean': tokens_per_sec_mean,
        'tokens_per_sec_min': tokens_per_sec_min,
        'tokens_per_sec_max': tokens_per_sec_max,
        'tokens_per_sec_std': tokens_per_sec_mean * (std_time / mean_time),
        'peak_memory_mb': peak_memory_mb,
        'current_memory_mb': current_memory_mb,
    }


def benchmark_model_config(
    config_name: str,
    config: Phase8Config,
    batch_sizes: List[int],
    seq_lens: List[int],
    device: str = 'cuda',
    num_iterations: int = 20,
):
    """特定の設定でベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    results = {
        'config_name': config_name,
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'num_heads': config.num_heads,
            'vocab_size': config.vocab_size,
            'use_bk_hyperbolic': config.use_bk_hyperbolic,
            'use_ar_ssm_fusion': config.use_ar_ssm_fusion,
            'enable_entailment_cones': config.enable_entailment_cones,
            'enable_persistent_homology': config.enable_persistent_homology,
            'enable_sheaf_attention': config.enable_sheaf_attention,
        },
        'benchmarks': []
    }
    
    try:
        # モデル作成
        print(f"Creating model...")
        
        # Tritonカーネルを強制的に有効化
        if TRITON_AVAILABLE:
            config.use_triton_kernel = True
            config.triton_kernel_version = 'fast'
            print("✓ Triton kernels enabled")
        else:
            config.use_triton_kernel = False
            print("✗ Triton kernels disabled (not available)")
        
        model = Phase8IntegratedModel(config)
        model = model.to(device)
        model.eval()
        
        # パラメータ数を計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        results['total_params'] = total_params
        results['total_params_millions'] = total_params / 1e6
        results['trainable_params'] = trainable_params
        results['triton_enabled'] = config.use_triton_kernel
        
        # 各設定でベンチマーク
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\n  Testing batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    # 入力データ作成
                    input_ids = torch.randint(
                        0, config.vocab_size,
                        (batch_size, seq_len),
                        device=device
                    )
                    
                    # ベンチマーク実行
                    bench_result = benchmark_throughput(
                        model, input_ids, num_iterations=num_iterations
                    )
                    
                    print(f"    Tokens/sec: {bench_result['tokens_per_sec_mean']:.0f} "
                          f"(±{bench_result['tokens_per_sec_std']:.0f})")
                    print(f"    Peak memory: {bench_result['peak_memory_mb']:.2f} MB")
                    
                    results['benchmarks'].append(bench_result)
                    
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if 'out of memory' in error_msg or 'cuda' in error_msg:
                        print(f"    SKIPPED: {str(e)[:100]}")
                        results['benchmarks'].append({
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'status': 'ERROR',
                            'error': str(e)[:200]
                        })
                        # メモリをクリア
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                    else:
                        print(f"    ERROR: {str(e)[:100]}")
                        results['benchmarks'].append({
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'status': 'ERROR',
                            'error': str(e)[:200]
                        })
                except Exception as e:
                    print(f"    UNEXPECTED ERROR: {str(e)[:100]}")
                    results['benchmarks'].append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'status': 'ERROR',
                        'error': str(e)[:200]
                    })
        
        results['status'] = 'SUCCESS'
        
    except Exception as e:
        print(f"ERROR: {e}")
        results['status'] = 'ERROR'
        results['error'] = str(e)
    
    finally:
        # クリーンアップ
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def main():
    """メインベンチマーク実行"""
    print("="*60)
    print("Phase 8 Comprehensive Benchmark")
    print("="*60)
    
    # Triton情報
    print(f"\nTriton Available: {TRITON_AVAILABLE}")
    if TRITON_AVAILABLE:
        print(f"Triton Version: {TRITON_VERSION}")
    else:
        print("WARNING: Triton not available - performance will be degraded")
        print("Install with: pip install triton")
    
    # GPU情報
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("\nWARNING: CUDA not available, using CPU")
        return  # CPUではベンチマーク不可
    
    device = 'cuda'
    
    # ベンチマーク設定
    configs_to_test = [
        # 小型モデル - 複数のシーケンス長でテスト
        {
            'name': 'Phase8_Small_83M_seq256',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=512,
                n_layers=8,
                n_seq=256,  # シーケンス長を明示的に設定
                num_heads=8,
                htt_rank=16,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2, 4, 8],
            'seq_lens': [256],
        },
        {
            'name': 'Phase8_Small_83M_seq512',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=512,
                n_layers=8,
                n_seq=512,
                num_heads=8,
                htt_rank=16,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2, 4],
            'seq_lens': [512],
        },
        {
            'name': 'Phase8_Small_83M_seq1024',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=512,
                n_layers=8,
                n_seq=1024,
                num_heads=8,
                htt_rank=16,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2],
            'seq_lens': [1024],
        },
        # 中型モデル
        {
            'name': 'Phase8_Medium_357M_seq512',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=1024,
                n_layers=16,
                n_seq=512,
                num_heads=16,
                htt_rank=32,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2, 4],
            'seq_lens': [512],
        },
        {
            'name': 'Phase8_Medium_357M_seq1024',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=1024,
                n_layers=16,
                n_seq=1024,
                num_heads=16,
                htt_rank=32,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2],
            'seq_lens': [1024],
        },
        # 大型モデル
        {
            'name': 'Phase8_Large_1.7B_seq512',
            'config': Phase8Config(
                vocab_size=50257,
                d_model=2048,
                n_layers=24,
                n_seq=512,
                num_heads=16,
                htt_rank=64,
                use_bk_hyperbolic=True,
                use_ar_ssm_fusion=True,
                enable_entailment_cones=False,
                enable_persistent_homology=False,
                enable_sheaf_attention=False,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
            ),
            'batch_sizes': [1, 2],
            'seq_lens': [512],
        },
    ]
    
    # 全結果を保存
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'triton_available': TRITON_AVAILABLE,
        'triton_version': TRITON_VERSION,
        'configs': []
    }
    
    # 各設定でベンチマーク
    for config_spec in configs_to_test:
        result = benchmark_model_config(
            config_name=config_spec['name'],
            config=config_spec['config'],
            batch_sizes=config_spec['batch_sizes'],
            seq_lens=config_spec['seq_lens'],
            device=device,
            num_iterations=20,
        )
        all_results['configs'].append(result)
    
    # 結果を保存
    output_file = 'results/benchmarks/phase8_comprehensive_benchmark.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # サマリーを表示
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for config_result in all_results['configs']:
        if config_result['status'] != 'SUCCESS':
            continue
        
        print(f"\n{config_result['config_name']}")
        print(f"  Parameters: {config_result['total_params_millions']:.2f}M")
        
        # 最良のスループットを見つける
        best_throughput = 0
        best_config = None
        
        for bench in config_result['benchmarks']:
            if 'tokens_per_sec_mean' in bench:
                if bench['tokens_per_sec_mean'] > best_throughput:
                    best_throughput = bench['tokens_per_sec_mean']
                    best_config = bench
        
        if best_config:
            print(f"  Best throughput: {best_throughput:.0f} tokens/sec")
            print(f"    (batch={best_config['batch_size']}, seq={best_config['seq_len']})")
            print(f"  Peak memory: {best_config['peak_memory_mb']:.2f} MB")


if __name__ == '__main__':
    main()
