#!/usr/bin/env python3
"""
Phase 8 Memory Optimizations Benchmark

タスク33.4: メモリ最適化のプロファイリング
- 各最適化によるメモリ削減量を測定
- 出力: results/benchmarks/phase8_memory_optimizations.json

Requirements: 33.1-33.4
"""

import json
import gc
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# メモリ最適化モジュールのインポート
try:
    from src.kernels.kv_cache_compression_triton import (
        KVCacheCompressor, CompressedKVCache
    )
    KV_COMPRESSION_AVAILABLE = True
except ImportError:
    KV_COMPRESSION_AVAILABLE = False

try:
    from src.kernels.memory_pool_triton import (
        HyperbolicMemoryPool, HyperbolicTensorAllocator
    )
    MEMORY_POOL_AVAILABLE = True
except ImportError:
    MEMORY_POOL_AVAILABLE = False

try:
    from src.kernels.prefetch_hyperbolic_triton import (
        PrefetchHyperbolicAttention, StreamingHyperbolicAttention
    )
    PREFETCH_AVAILABLE = True
except ImportError:
    PREFETCH_AVAILABLE = False


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
    }


def clear_memory(device: torch.device):
    """メモリをクリア"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_memory_usage(
    fn,
    *args,
    device: torch.device = None,
) -> Dict[str, float]:
    """メモリ使用量を測定"""
    device = device or get_device()
    clear_memory(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # 実行
    result = fn(*args)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    peak_memory_mb = 0.0
    allocated_mb = 0.0
    
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return {
        "peak_memory_mb": peak_memory_mb,
        "allocated_mb": allocated_mb,
        "result": result,
    }


class MemoryOptimizationsBenchmark:
    """メモリ最適化ベンチマーク"""
    
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
    
    def benchmark_kv_compression(self, seq_len: int) -> Dict[str, Any]:
        """KVキャッシュ圧縮のメモリ削減を測定"""
        if not KV_COMPRESSION_AVAILABLE:
            return {"error": "Not available"}
        
        try:
            # 元のKVキャッシュサイズ
            k = torch.randn(
                self.batch_size, self.num_heads, seq_len, self.head_dim,
                device=self.device, dtype=torch.float16
            )
            v = torch.randn(
                self.batch_size, self.num_heads, seq_len, self.head_dim,
                device=self.device, dtype=torch.float16
            )
            
            original_size_mb = (k.numel() + v.numel()) * k.element_size() / (1024 * 1024)
            
            # 圧縮
            k_c, v_c, metadata = KVCacheCompressor.compress(k, v)
            
            compressed_size_mb = (
                k_c.numel() * k_c.element_size() +
                v_c.numel() * v_c.element_size() +
                sum(m.numel() * m.element_size() for m in metadata.values())
            ) / (1024 * 1024)
            
            # 展開精度
            k_decompressed, v_decompressed = KVCacheCompressor.decompress(k_c, v_c, metadata)
            
            k_error = (k.float() - k_decompressed).abs().mean().item()
            v_error = (v.float() - v_decompressed).abs().mean().item()
            
            return {
                "original_size_mb": original_size_mb,
                "compressed_size_mb": compressed_size_mb,
                "compression_ratio": original_size_mb / compressed_size_mb,
                "memory_savings_percent": (1 - compressed_size_mb / original_size_mb) * 100,
                "k_reconstruction_error": k_error,
                "v_reconstruction_error": v_error,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_memory_pool(self, seq_len: int) -> Dict[str, Any]:
        """メモリプールの効率を測定"""
        if not MEMORY_POOL_AVAILABLE:
            return {"error": "Not available"}
        
        try:
            pool = HyperbolicMemoryPool(
                initial_size_mb=64,
                max_size_mb=256,
                device=self.device,
            )
            
            shape = (self.batch_size, self.num_heads, seq_len, self.head_dim)
            
            # 複数回のアロケーション/デアロケーション
            num_ops = 100
            
            # プールなしの場合
            clear_memory(self.device)
            start_time = time.perf_counter()
            
            for _ in range(num_ops):
                tensor = torch.empty(shape, device=self.device, dtype=torch.float16)
                del tensor
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            no_pool_time = time.perf_counter() - start_time
            
            # プールありの場合
            clear_memory(self.device)
            start_time = time.perf_counter()
            
            for _ in range(num_ops):
                tensor = pool.allocate(shape)
                pool.deallocate(tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            pool_time = time.perf_counter() - start_time
            
            stats = pool.get_stats()
            
            return {
                "no_pool_time_ms": no_pool_time * 1000,
                "pool_time_ms": pool_time * 1000,
                "speedup": no_pool_time / pool_time if pool_time > 0 else 0,
                "cache_hit_rate": stats['cache_hits'] / max(1, stats['allocations']),
                "fragmentation_ratio": stats['fragmentation_ratio'],
                "allocation_time_us": (pool_time / num_ops) * 1e6,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_streaming_attention(self, seq_len: int) -> Dict[str, Any]:
        """ストリーミングアテンションのメモリ効率を測定"""
        if not PREFETCH_AVAILABLE:
            return {"error": "Not available"}
        
        try:
            # 標準アテンション
            clear_memory(self.device)
            
            q = torch.randn(
                self.batch_size, self.num_heads, seq_len, self.head_dim,
                device=self.device
            )
            k = torch.randn(
                self.batch_size, self.num_heads, seq_len, self.head_dim,
                device=self.device
            )
            v = torch.randn(
                self.batch_size, self.num_heads, seq_len, self.head_dim,
                device=self.device
            )
            
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            # 標準アテンション（フルマテリアライズ）
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            out_standard = torch.matmul(attn_weights, v)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                standard_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                standard_memory_mb = 0
            
            del attn_weights, out_standard
            clear_memory(self.device)
            
            # ストリーミングアテンション
            streaming_module = StreamingHyperbolicAttention(
                self.d_model, self.num_heads, curvature=1.0, use_triton=False
            ).to(self.device)
            
            x = torch.randn(self.batch_size, seq_len, self.d_model, device=self.device)
            
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            out_streaming = streaming_module(x)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                streaming_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                streaming_memory_mb = 0
            
            return {
                "standard_memory_mb": standard_memory_mb,
                "streaming_memory_mb": streaming_memory_mb,
                "memory_savings_percent": (1 - streaming_memory_mb / standard_memory_mb) * 100 if standard_memory_mb > 0 else 0,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_compressed_kv_cache(self, seq_len: int) -> Dict[str, Any]:
        """圧縮KVキャッシュモジュールのベンチマーク"""
        if not KV_COMPRESSION_AVAILABLE:
            return {"error": "Not available"}
        
        try:
            cache = CompressedKVCache(
                max_seq_len=seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=self.device,
            )
            
            # キャッシュに追加
            chunk_size = 64
            for start_pos in range(0, seq_len, chunk_size):
                k = torch.randn(
                    1, self.num_heads, chunk_size, self.head_dim,
                    device=self.device
                )
                v = torch.randn(
                    1, self.num_heads, chunk_size, self.head_dim,
                    device=self.device
                )
                cache.update(k, v, start_pos)
            
            # 圧縮率
            compression_ratio = cache.get_compression_ratio()
            
            # 取得時間
            start_time = time.perf_counter()
            k_out, v_out = cache.get()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            get_time_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "compression_ratio": compression_ratio,
                "get_time_ms": get_time_ms,
                "cache_length": cache.current_len,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096, 8192],
    ) -> Dict[str, Any]:
        """フルベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Memory Optimizations Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"\nOptimizations available:")
        print(f"  KV Compression: {KV_COMPRESSION_AVAILABLE}")
        print(f"  Memory Pool: {MEMORY_POOL_AVAILABLE}")
        print(f"  Prefetch/Streaming: {PREFETCH_AVAILABLE}")
        print(f"{'='*60}\n")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_info": get_gpu_info(),
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
            },
            "kv_compression": {},
            "memory_pool": {},
            "streaming_attention": {},
            "compressed_kv_cache": {},
        }
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # KV圧縮
            kv_result = self.benchmark_kv_compression(seq_len)
            results["kv_compression"][str(seq_len)] = kv_result
            if "error" not in kv_result:
                print(f"  KV Compression: {kv_result['compression_ratio']:.2f}x, "
                      f"{kv_result['memory_savings_percent']:.1f}% savings")
            else:
                print(f"  KV Compression: {kv_result['error']}")
            
            # メモリプール
            pool_result = self.benchmark_memory_pool(seq_len)
            results["memory_pool"][str(seq_len)] = pool_result
            if "error" not in pool_result:
                print(f"  Memory Pool: {pool_result['speedup']:.2f}x speedup, "
                      f"{pool_result['allocation_time_us']:.2f} us/alloc")
            else:
                print(f"  Memory Pool: {pool_result['error']}")
            
            # ストリーミングアテンション
            streaming_result = self.benchmark_streaming_attention(seq_len)
            results["streaming_attention"][str(seq_len)] = streaming_result
            if "error" not in streaming_result:
                print(f"  Streaming Attention: {streaming_result['memory_savings_percent']:.1f}% savings")
            else:
                print(f"  Streaming Attention: {streaming_result['error']}")
            
            # 圧縮KVキャッシュ
            cache_result = self.benchmark_compressed_kv_cache(seq_len)
            results["compressed_kv_cache"][str(seq_len)] = cache_result
            if "error" not in cache_result:
                print(f"  Compressed KV Cache: {cache_result['compression_ratio']:.2f}x")
            else:
                print(f"  Compressed KV Cache: {cache_result['error']}")
            
            clear_memory(self.device)
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        # 平均圧縮率
        kv_ratios = [
            r['compression_ratio'] for r in results["kv_compression"].values()
            if isinstance(r, dict) and 'compression_ratio' in r
        ]
        if kv_ratios:
            print(f"Average KV compression ratio: {sum(kv_ratios)/len(kv_ratios):.2f}x")
        
        # 平均メモリプールスピードアップ
        pool_speedups = [
            r['speedup'] for r in results["memory_pool"].values()
            if isinstance(r, dict) and 'speedup' in r
        ]
        if pool_speedups:
            print(f"Average memory pool speedup: {sum(pool_speedups)/len(pool_speedups):.2f}x")
        
        results["summary"] = {
            "avg_kv_compression_ratio": sum(kv_ratios)/len(kv_ratios) if kv_ratios else 0,
            "avg_pool_speedup": sum(pool_speedups)/len(pool_speedups) if pool_speedups else 0,
        }
        
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
        description="Phase 8 Memory Optimizations Benchmark"
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
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_memory_optimizations.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = MemoryOptimizationsBenchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
    
    benchmark.run_benchmark(seq_lengths=args.seq_lengths)
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
