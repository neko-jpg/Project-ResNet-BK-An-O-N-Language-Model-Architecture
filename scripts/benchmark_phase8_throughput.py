#!/usr/bin/env python3
"""
Phase 8 Throughput Benchmark Script

タスク30.1: Phase 8のスループットベンチマーク
- seq=1024, 2048, 4096, 8192でのtokens/sec測定
- Phase 7ベースラインとの比較
- 目標: 2x throughput improvement

Requirements: 30.1, 30.2
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# Phase 8モジュールのインポート
try:
    from src.models.phase8 import (
        FlashHyperbolicAttention,
        HyperbolicSSM,
        LinearHyperbolicAttention,
        AdaptiveHyperbolicComputation,
    )
    PHASE8_AVAILABLE = True
except ImportError:
    PHASE8_AVAILABLE = False
    print("Warning: Phase 8 modules not fully available")

# Phase 7モジュールのインポート
try:
    from src.models.phase7 import HyperbolicAttention
    PHASE7_AVAILABLE = True
except ImportError:
    PHASE7_AVAILABLE = False
    print("Warning: Phase 7 modules not available")


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def warmup_gpu(device: torch.device, iterations: int = 10):
    """GPUウォームアップ"""
    if device.type == "cuda":
        x = torch.randn(1024, 1024, device=device)
        for _ in range(iterations):
            _ = torch.matmul(x, x)
        torch.cuda.synchronize()


def measure_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    モデルのスループットを測定
    
    Returns:
        Dict with tokens_per_sec, latency_ms, memory_mb
    """
    model = model.to(device)
    model.eval()
    
    # 入力データ生成
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # メモリ使用量測定
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # スループット測定
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # 結果計算
    total_time = end_time - start_time
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_sec = total_tokens / total_time
    latency_ms = (total_time / num_iterations) * 1000
    
    memory_mb = 0.0
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "total_time_sec": total_time,
        "num_iterations": num_iterations,
    }


class Phase8ThroughputBenchmark:
    """Phase 8スループットベンチマーク"""
    
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
    
    def create_phase8_model(self, seq_len: int) -> nn.Module:
        """Phase 8モデルを作成"""
        if not PHASE8_AVAILABLE:
            # フォールバック: シンプルなアテンション
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        
        # Flash Hyperbolic Attentionを使用
        try:
            return FlashHyperbolicAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                curvature=1.0,
            )
        except Exception:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
    
    def create_phase7_model(self, seq_len: int) -> nn.Module:
        """Phase 7モデルを作成（ベースライン）"""
        if not PHASE7_AVAILABLE:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        
        try:
            return HyperbolicAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                curvature=1.0,
            )
        except Exception:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_iterations: int = 50,
    ) -> Dict[str, Any]:
        """ベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Throughput Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"{'='*60}\n")
        
        # GPUウォームアップ
        warmup_gpu(self.device)
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_iterations": num_iterations,
            },
            "phase8": {},
            "phase7_baseline": {},
            "comparison": {},
        }
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # Phase 8ベンチマーク
            try:
                phase8_model = self.create_phase8_model(seq_len)
                phase8_results = measure_throughput(
                    phase8_model,
                    self.batch_size,
                    seq_len,
                    self.d_model,
                    self.device,
                    num_iterations,
                )
                results["phase8"][str(seq_len)] = phase8_results
                print(f"  Phase 8: {phase8_results['tokens_per_sec']:.0f} tokens/sec")
                print(f"           {phase8_results['latency_ms']:.2f} ms latency")
                print(f"           {phase8_results['memory_mb']:.1f} MB memory")
            except Exception as e:
                print(f"  Phase 8: Error - {e}")
                results["phase8"][str(seq_len)] = {"error": str(e)}
            
            # Phase 7ベースラインベンチマーク
            try:
                phase7_model = self.create_phase7_model(seq_len)
                phase7_results = measure_throughput(
                    phase7_model,
                    self.batch_size,
                    seq_len,
                    self.d_model,
                    self.device,
                    num_iterations,
                )
                results["phase7_baseline"][str(seq_len)] = phase7_results
                print(f"  Phase 7: {phase7_results['tokens_per_sec']:.0f} tokens/sec")
                print(f"           {phase7_results['latency_ms']:.2f} ms latency")
                print(f"           {phase7_results['memory_mb']:.1f} MB memory")
            except Exception as e:
                print(f"  Phase 7: Error - {e}")
                results["phase7_baseline"][str(seq_len)] = {"error": str(e)}
            
            # 比較計算
            if (str(seq_len) in results["phase8"] and 
                str(seq_len) in results["phase7_baseline"] and
                "error" not in results["phase8"][str(seq_len)] and
                "error" not in results["phase7_baseline"][str(seq_len)]):
                
                p8_tps = results["phase8"][str(seq_len)]["tokens_per_sec"]
                p7_tps = results["phase7_baseline"][str(seq_len)]["tokens_per_sec"]
                speedup = p8_tps / p7_tps if p7_tps > 0 else 0
                
                p8_mem = results["phase8"][str(seq_len)]["memory_mb"]
                p7_mem = results["phase7_baseline"][str(seq_len)]["memory_mb"]
                mem_reduction = (1 - p8_mem / p7_mem) * 100 if p7_mem > 0 else 0
                
                results["comparison"][str(seq_len)] = {
                    "speedup": speedup,
                    "memory_reduction_percent": mem_reduction,
                    "target_met": speedup >= 2.0,
                }
                
                print(f"  Speedup: {speedup:.2f}x (target: 2.0x)")
                print(f"  Memory reduction: {mem_reduction:.1f}%")
            
            # メモリクリア
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        targets_met = sum(
            1 for v in results["comparison"].values() 
            if isinstance(v, dict) and v.get("target_met", False)
        )
        total_tests = len(results["comparison"])
        
        results["summary"] = {
            "targets_met": targets_met,
            "total_tests": total_tests,
            "success_rate": targets_met / total_tests if total_tests > 0 else 0,
            "average_speedup": sum(
                v["speedup"] for v in results["comparison"].values() 
                if isinstance(v, dict) and "speedup" in v
            ) / total_tests if total_tests > 0 else 0,
        }
        
        print(f"Targets met: {targets_met}/{total_tests}")
        print(f"Average speedup: {results['summary']['average_speedup']:.2f}x")
        
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
        description="Phase 8 Throughput Benchmark"
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
        "--num-iterations", type=int, default=50,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_throughput_benchmark.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = Phase8ThroughputBenchmark(
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
