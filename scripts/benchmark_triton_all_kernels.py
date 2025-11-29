#!/usr/bin/env python3
"""
Phase 8 All Triton Kernels Benchmark

タスク32.5: 全Tritonカーネルのベンチマーク
- fused, quantized, sparse, register-tiledカーネルの比較
- 出力: results/benchmarks/phase8_triton_all_kernels.json

Requirements: 32.1-32.5
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# カーネルのインポート
KERNELS_AVAILABLE = {}

try:
    from src.kernels.fused_ln_hyperbolic_triton import FusedLNHyperbolic
    KERNELS_AVAILABLE['fused_ln'] = True
except ImportError:
    KERNELS_AVAILABLE['fused_ln'] = False

try:
    from src.kernels.quantized_hyperbolic_triton import QuantizedHyperbolicAttention
    KERNELS_AVAILABLE['quantized'] = True
except ImportError:
    KERNELS_AVAILABLE['quantized'] = False

try:
    from src.kernels.sparse_hyperbolic_triton import SparseHyperbolicAttention
    KERNELS_AVAILABLE['sparse'] = True
except ImportError:
    KERNELS_AVAILABLE['sparse'] = False

try:
    from src.kernels.register_tiled_distance_triton import RegisterTiledHyperbolicAttention
    KERNELS_AVAILABLE['register_tiled'] = True
except ImportError:
    KERNELS_AVAILABLE['register_tiled'] = False

try:
    from src.kernels.kv_cache_compression_triton import CompressedKVCache, KVCacheCompressor
    KERNELS_AVAILABLE['kv_compression'] = True
except ImportError:
    KERNELS_AVAILABLE['kv_compression'] = False

try:
    from src.kernels.memory_pool_triton import HyperbolicMemoryPool
    KERNELS_AVAILABLE['memory_pool'] = True
except ImportError:
    KERNELS_AVAILABLE['memory_pool'] = False

try:
    from src.kernels.prefetch_hyperbolic_triton import PrefetchHyperbolicAttention
    KERNELS_AVAILABLE['prefetch'] = True
except ImportError:
    KERNELS_AVAILABLE['prefetch'] = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count,
    }


def warmup_gpu(device: torch.device, iterations: int = 10):
    """GPUウォームアップ"""
    if device.type == "cuda":
        x = torch.randn(1024, 1024, device=device)
        for _ in range(iterations):
            _ = torch.matmul(x, x)
        torch.cuda.synchronize()


def measure_kernel_time(
    fn,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: torch.device = None,
) -> Dict[str, float]:
    """カーネル実行時間を測定"""
    device = device or get_device()
    
    # ウォームアップ
    for _ in range(warmup_iterations):
        try:
            _ = fn(*args)
            if device.type == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass
    
    # メモリリセット
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # 測定
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_iter_ms = (total_time / num_iterations) * 1000
    
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return {
        "time_ms": time_per_iter_ms,
        "total_time_sec": total_time,
        "peak_memory_mb": peak_memory_mb,
    }


class AllKernelsBenchmark:
    """全Tritonカーネルベンチマーク"""
    
    def __init__(
        self,
        batch_size: int = 4,
        d_model: int = 512,
        num_heads: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device or get_device()
        self.results: Dict[str, Any] = {}
    
    def benchmark_fused_ln(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Fused LayerNorm + Hyperbolicベンチマーク"""
        if not KERNELS_AVAILABLE['fused_ln']:
            return {"error": "Not available"}
        
        try:
            module = FusedLNHyperbolic(self.d_model, curvature=1.0).to(self.device)
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            return measure_kernel_time(
                module, x,
                num_iterations=num_iterations,
                device=self.device,
            )
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_quantized(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Quantized Attentionベンチマーク"""
        if not KERNELS_AVAILABLE['quantized']:
            return {"error": "Not available"}
        
        try:
            module = QuantizedHyperbolicAttention(
                self.d_model, self.num_heads, curvature=1.0, quantization_bits=8
            ).to(self.device)
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            return measure_kernel_time(
                module, x,
                num_iterations=num_iterations,
                device=self.device,
            )
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_sparse(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Sparse Attentionベンチマーク"""
        if not KERNELS_AVAILABLE['sparse']:
            return {"error": "Not available"}
        
        try:
            module = SparseHyperbolicAttention(
                self.d_model, self.num_heads, curvature=1.0, sparsity_ratio=0.9
            ).to(self.device)
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            return measure_kernel_time(
                module, x,
                num_iterations=num_iterations,
                device=self.device,
            )
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_register_tiled(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Register-Tiled Attentionベンチマーク"""
        if not KERNELS_AVAILABLE['register_tiled']:
            return {"error": "Not available"}
        
        try:
            module = RegisterTiledHyperbolicAttention(
                self.d_model, self.num_heads, curvature=1.0
            ).to(self.device)
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            return measure_kernel_time(
                module, x,
                num_iterations=num_iterations,
                device=self.device,
            )
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_prefetch(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Prefetch Attentionベンチマーク"""
        if not KERNELS_AVAILABLE['prefetch']:
            return {"error": "Not available"}
        
        try:
            module = PrefetchHyperbolicAttention(
                self.d_model, self.num_heads, curvature=1.0
            ).to(self.device)
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            return measure_kernel_time(
                module, x,
                num_iterations=num_iterations,
                device=self.device,
            )
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_kv_compression(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """KV Cache Compressionベンチマーク"""
        if not KERNELS_AVAILABLE['kv_compression']:
            return {"error": "Not available"}
        
        try:
            k = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            v = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            
            def compress_decompress():
                k_c, v_c, meta = KVCacheCompressor.compress(k, v)
                return KVCacheCompressor.decompress(k_c, v_c, meta)
            
            result = measure_kernel_time(
                compress_decompress,
                num_iterations=num_iterations,
                device=self.device,
            )
            
            # 圧縮率を計算
            k_c, v_c, _ = KVCacheCompressor.compress(k, v)
            original_size = k.numel() * k.element_size() + v.numel() * v.element_size()
            compressed_size = k_c.numel() * k_c.element_size() + v_c.numel() * v_c.element_size()
            result['compression_ratio'] = original_size / compressed_size
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_memory_pool(self, seq_len: int, num_iterations: int) -> Dict[str, Any]:
        """Memory Poolベンチマーク"""
        if not KERNELS_AVAILABLE['memory_pool']:
            return {"error": "Not available"}
        
        try:
            pool = HyperbolicMemoryPool(
                initial_size_mb=64,
                max_size_mb=256,
                device=self.device,
            )
            
            shape = (self.batch_size, self.num_heads, seq_len, self.head_dim)
            
            # アロケーション時間を測定
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                tensor = pool.allocate(shape)
                pool.deallocate(tensor)
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            time_per_iter_us = (total_time / num_iterations) * 1e6
            
            stats = pool.get_stats()
            
            return {
                "allocation_time_us": time_per_iter_us,
                "cache_hit_rate": stats['cache_hits'] / max(1, stats['allocations']),
                "fragmentation_ratio": stats['fragmentation_ratio'],
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096],
        num_iterations: int = 50,
    ) -> Dict[str, Any]:
        """フルベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 All Triton Kernels Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"\nKernels available:")
        for name, available in KERNELS_AVAILABLE.items():
            print(f"  {name}: {'Yes' if available else 'No'}")
        print(f"{'='*60}\n")
        
        warmup_gpu(self.device)
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_info": get_gpu_info(),
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_iterations": num_iterations,
                "kernels_available": KERNELS_AVAILABLE,
            },
            "benchmarks": {},
        }
        
        kernels = [
            ("fused_ln", self.benchmark_fused_ln),
            ("quantized", self.benchmark_quantized),
            ("sparse", self.benchmark_sparse),
            ("register_tiled", self.benchmark_register_tiled),
            ("prefetch", self.benchmark_prefetch),
            ("kv_compression", self.benchmark_kv_compression),
            ("memory_pool", self.benchmark_memory_pool),
        ]
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            results["benchmarks"][str(seq_len)] = {}
            
            for kernel_name, benchmark_fn in kernels:
                try:
                    result = benchmark_fn(seq_len, num_iterations)
                    results["benchmarks"][str(seq_len)][kernel_name] = result
                    
                    if "error" not in result:
                        if "time_ms" in result:
                            print(f"  {kernel_name}: {result['time_ms']:.3f} ms")
                        elif "allocation_time_us" in result:
                            print(f"  {kernel_name}: {result['allocation_time_us']:.2f} us")
                    else:
                        print(f"  {kernel_name}: {result['error']}")
                except Exception as e:
                    print(f"  {kernel_name}: Error - {e}")
                    results["benchmarks"][str(seq_len)][kernel_name] = {"error": str(e)}
            
            # メモリクリア
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        # 各カーネルの平均時間
        for kernel_name, _ in kernels:
            times = []
            for seq_len in seq_lengths:
                result = results["benchmarks"].get(str(seq_len), {}).get(kernel_name, {})
                if "time_ms" in result:
                    times.append(result["time_ms"])
            
            if times:
                avg_time = sum(times) / len(times)
                print(f"{kernel_name}: avg {avg_time:.3f} ms")
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """結果をJSONファイルに保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 All Triton Kernels Benchmark"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--d-model", type=int, default=512,
        help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+",
        default=[1024, 2048, 4096],
        help="Sequence lengths to benchmark"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=50,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_triton_all_kernels.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = AllKernelsBenchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
    
    benchmark.run_benchmark(
        seq_lengths=args.seq_lengths,
        num_iterations=args.num_iterations,
    )
    
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
