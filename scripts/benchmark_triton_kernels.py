#!/usr/bin/env python3
"""
Phase 8 Triton Kernels Benchmark Script

タスク30.3: Tritonカーネルの比較ベンチマーク
- Flash, Enhanced, Quantizedカーネルの比較
- FLOPS利用率の測定
- 目標: 70%+ FLOPS利用率

Requirements: 31.4, 26.4
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# Tritonカーネルのインポート
TRITON_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    pass

# Phase 8カーネルのインポート
try:
    from src.kernels.flash_hyperbolic_triton import flash_hyperbolic_attention
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

try:
    from src.kernels.enhanced_hyperbolic_triton import enhanced_hyperbolic_attention
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from src.models.phase8.quantization import INT8QuantizedKernel
    QUANTIZED_AVAILABLE = True
except ImportError:
    QUANTIZED_AVAILABLE = False


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
        "max_threads_per_block": props.max_threads_per_block,
    }


def calculate_theoretical_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
) -> float:
    """
    理論FLOPSを計算
    
    Attention: 2 * B * H * N * N * D/H (QK^T) + 2 * B * H * N * N * D/H (softmax*V)
    """
    head_dim = d_model // num_heads
    # QK^T: B * H * N * N * D/H * 2 (multiply-add)
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    # Softmax: ~5 * B * H * N * N (exp, sum, div)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    # Attention * V: B * H * N * N * D/H * 2
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    return qk_flops + softmax_flops + av_flops


def measure_kernel_performance(
    kernel_fn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, Any]:
    """
    カーネルのパフォーマンスを測定
    
    Returns:
        Dict with time_ms, flops, flops_utilization
    """
    head_dim = d_model // num_heads
    
    # 入力データ生成
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # ウォームアップ
    for _ in range(warmup_iterations):
        try:
            _ = kernel_fn(q, k, v)
            torch.cuda.synchronize()
        except Exception:
            pass
    
    # 測定
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = kernel_fn(q, k, v)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_iter_ms = (total_time / num_iterations) * 1000
    
    # FLOPS計算
    theoretical_flops = calculate_theoretical_flops(batch_size, seq_len, d_model, num_heads)
    achieved_flops = theoretical_flops / (time_per_iter_ms / 1000)
    
    # GPU理論ピークFLOPS（RTX 3080: ~29.8 TFLOPS FP16）
    # 実際のGPUに応じて調整
    gpu_peak_tflops = 29.8  # RTX 3080
    flops_utilization = (achieved_flops / (gpu_peak_tflops * 1e12)) * 100
    
    return {
        "time_ms": time_per_iter_ms,
        "theoretical_flops": theoretical_flops,
        "achieved_flops": achieved_flops,
        "achieved_tflops": achieved_flops / 1e12,
        "flops_utilization_percent": flops_utilization,
        "target_met": flops_utilization >= 70,
    }


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorchリファレンス実装"""
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class TritonKernelsBenchmark:
    """Tritonカーネルベンチマーク"""
    
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
        self.device = device or get_device()
        self.results: Dict[str, Any] = {}
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096],
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """ベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Triton Kernels Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"Triton available: {TRITON_AVAILABLE}")
        print(f"{'='*60}\n")
        
        gpu_info = get_gpu_info()
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_info": gpu_info,
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_iterations": num_iterations,
            },
            "pytorch_baseline": {},
            "flash_hyperbolic": {},
            "enhanced_hyperbolic": {},
            "quantized_int8": {},
            "comparison": {},
        }
        
        kernels = [
            ("pytorch_baseline", pytorch_attention, True),
            ("flash_hyperbolic", flash_hyperbolic_attention if FLASH_AVAILABLE else None, FLASH_AVAILABLE),
            ("enhanced_hyperbolic", enhanced_hyperbolic_attention if ENHANCED_AVAILABLE else None, ENHANCED_AVAILABLE),
        ]
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            for kernel_name, kernel_fn, available in kernels:
                if not available or kernel_fn is None:
                    print(f"  {kernel_name}: Not available")
                    results[kernel_name][str(seq_len)] = {"error": "Not available"}
                    continue
                
                try:
                    perf = measure_kernel_performance(
                        kernel_fn,
                        self.batch_size,
                        seq_len,
                        self.d_model,
                        self.num_heads,
                        self.device,
                        num_iterations,
                    )
                    results[kernel_name][str(seq_len)] = perf
                    
                    print(f"  {kernel_name}:")
                    print(f"    Time: {perf['time_ms']:.2f} ms")
                    print(f"    TFLOPS: {perf['achieved_tflops']:.2f}")
                    print(f"    Utilization: {perf['flops_utilization_percent']:.1f}%")
                except Exception as e:
                    print(f"  {kernel_name}: Error - {e}")
                    results[kernel_name][str(seq_len)] = {"error": str(e)}
            
            # 比較計算
            baseline = results["pytorch_baseline"].get(str(seq_len), {})
            if "time_ms" in baseline:
                comparison = {"seq_len": seq_len}
                
                for kernel_name in ["flash_hyperbolic", "enhanced_hyperbolic"]:
                    kernel_result = results[kernel_name].get(str(seq_len), {})
                    if "time_ms" in kernel_result:
                        speedup = baseline["time_ms"] / kernel_result["time_ms"]
                        comparison[f"{kernel_name}_speedup"] = speedup
                
                results["comparison"][str(seq_len)] = comparison
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        # 平均FLOPS利用率
        for kernel_name in ["flash_hyperbolic", "enhanced_hyperbolic"]:
            utilizations = [
                v["flops_utilization_percent"]
                for v in results[kernel_name].values()
                if isinstance(v, dict) and "flops_utilization_percent" in v
            ]
            if utilizations:
                avg_util = sum(utilizations) / len(utilizations)
                print(f"{kernel_name} avg utilization: {avg_util:.1f}%")
        
        # 平均スピードアップ
        for kernel_name in ["flash_hyperbolic", "enhanced_hyperbolic"]:
            speedups = [
                v.get(f"{kernel_name}_speedup", 0)
                for v in results["comparison"].values()
                if isinstance(v, dict)
            ]
            speedups = [s for s in speedups if s > 0]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"{kernel_name} avg speedup: {avg_speedup:.2f}x")
        
        results["summary"] = {
            "flash_avg_utilization": sum(
                v["flops_utilization_percent"]
                for v in results["flash_hyperbolic"].values()
                if isinstance(v, dict) and "flops_utilization_percent" in v
            ) / max(1, len([v for v in results["flash_hyperbolic"].values() if isinstance(v, dict) and "flops_utilization_percent" in v])),
            "enhanced_avg_utilization": sum(
                v["flops_utilization_percent"]
                for v in results["enhanced_hyperbolic"].values()
                if isinstance(v, dict) and "flops_utilization_percent" in v
            ) / max(1, len([v for v in results["enhanced_hyperbolic"].values() if isinstance(v, dict) and "flops_utilization_percent" in v])),
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
        description="Phase 8 Triton Kernels Benchmark"
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
        "--num-iterations", type=int, default=100,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_triton_benchmark.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = TritonKernelsBenchmark(
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
