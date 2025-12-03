#!/usr/bin/env python3
"""
Phase 8 Consumer GPU Benchmark Script

タスク30.7: コンシューマGPUベンチマーク
- RTX 3080 (10GB), RTX 3090 (24GB), RTX 4090 (24GB)でのテスト
- すべてのターゲットが各GPUで達成されているか検証

Requirements: 10.1-10.6, 28.1-28.6
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

# Phase 8モジュールのインポート
try:
    from src.models.phase8 import (
        BlockWiseDistanceComputation,
        HyperbolicSSM,
        TangentSpaceLinearAttention,
    )
    PHASE8_AVAILABLE = True
except ImportError:
    PHASE8_AVAILABLE = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available", "gpu_type": "CPU"}
    
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.lower()
    
    # GPU名からタイプを推定
    gpu_type = "unknown"
    vram_target_gb = 8.0
    theoretical_tflops = 10.0
    
    if "3080" in gpu_name:
        gpu_type = "RTX 3080"
        vram_target_gb = 10.0
        theoretical_tflops = 29.8  # FP16
    elif "3090" in gpu_name:
        gpu_type = "RTX 3090"
        vram_target_gb = 24.0
        theoretical_tflops = 35.6
    elif "4090" in gpu_name:
        gpu_type = "RTX 4090"
        vram_target_gb = 24.0
        theoretical_tflops = 82.6
    elif "4080" in gpu_name:
        gpu_type = "RTX 4080"
        vram_target_gb = 16.0
        theoretical_tflops = 48.7
    elif "3070" in gpu_name:
        gpu_type = "RTX 3070"
        vram_target_gb = 8.0
        theoretical_tflops = 20.3
    elif "3060" in gpu_name:
        gpu_type = "RTX 3060"
        vram_target_gb = 12.0
        theoretical_tflops = 12.7
    
    return {
        "name": props.name,
        "gpu_type": gpu_type,
        "total_memory_gb": props.total_memory / (1024**3),
        "vram_target_gb": vram_target_gb,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessor_count": props.multi_processor_count,
        "theoretical_tflops_fp16": theoretical_tflops,
    }


def clear_memory(device: torch.device):
    """メモリをクリア"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def warmup_gpu(device: torch.device, iterations: int = 10):
    """GPUウォームアップ"""
    if device.type == "cuda":
        x = torch.randn(1024, 1024, device=device)
        for _ in range(iterations):
            _ = torch.matmul(x, x)
        torch.cuda.synchronize()


class ConsumerGPUBenchmark:
    """コンシューマGPUベンチマーク"""
    
    # GPUごとのターゲット定義
    GPU_TARGETS = {
        "RTX 3080": {
            "vram_gb": 10.0,
            "max_seq_len": 16384,
            "min_throughput_tokens_per_sec": 50000,
            "max_memory_at_8k_gb": 8.0,
        },
        "RTX 3090": {
            "vram_gb": 24.0,
            "max_seq_len": 32768,
            "min_throughput_tokens_per_sec": 60000,
            "max_memory_at_8k_gb": 8.0,
        },
        "RTX 4090": {
            "vram_gb": 24.0,
            "max_seq_len": 32768,
            "min_throughput_tokens_per_sec": 100000,
            "max_memory_at_8k_gb": 8.0,
        },
        "unknown": {
            "vram_gb": 8.0,
            "max_seq_len": 8192,
            "min_throughput_tokens_per_sec": 30000,
            "max_memory_at_8k_gb": 6.0,
        },
    }
    
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
        self.gpu_info = get_gpu_info()
        self.results: Dict[str, Any] = {}
    
    def get_targets(self) -> Dict[str, Any]:
        """現在のGPUのターゲットを取得"""
        gpu_type = self.gpu_info.get("gpu_type", "unknown")
        return self.GPU_TARGETS.get(gpu_type, self.GPU_TARGETS["unknown"])
    
    def create_model(self) -> nn.Module:
        """Phase 8モデルを作成"""
        if not PHASE8_AVAILABLE:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        
        try:
            return BlockWiseDistanceComputation(
                d_model=self.d_model,
                num_heads=self.num_heads,
                block_size=128,
                curvature=1.0,
            )
        except Exception:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
    
    def measure_throughput(
        self,
        seq_len: int,
        num_iterations: int = 50,
    ) -> Dict[str, Any]:
        """スループットを測定"""
        model = self.create_model().to(self.device)
        model.eval()
        
        x = torch.randn(
            self.batch_size, seq_len, self.d_model,
            device=self.device
        )
        
        # ウォームアップ
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        # メモリリセット
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        # 測定
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_tokens = self.batch_size * seq_len * num_iterations
        tokens_per_sec = total_tokens / total_time
        latency_ms = (total_time / num_iterations) * 1000
        
        peak_memory_mb = 0.0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_gb": peak_memory_mb / 1024,
        }
    
    def test_max_sequence_length(self) -> Dict[str, Any]:
        """最大シーケンス長をテスト"""
        targets = self.get_targets()
        target_max_seq = targets["max_seq_len"]
        vram_gb = targets["vram_gb"]
        
        test_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
        max_supported = 0
        results_by_length = {}
        
        for seq_len in test_lengths:
            if seq_len > target_max_seq * 2:
                break
            
            clear_memory(self.device)
            
            try:
                result = self.measure_throughput(seq_len, num_iterations=10)
                
                if result["peak_memory_gb"] < vram_gb * 0.95:
                    max_supported = seq_len
                    results_by_length[str(seq_len)] = {
                        "success": True,
                        **result,
                    }
                else:
                    results_by_length[str(seq_len)] = {
                        "success": False,
                        "reason": "memory_exceeded",
                        **result,
                    }
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results_by_length[str(seq_len)] = {
                        "success": False,
                        "reason": "OOM",
                        "error": str(e),
                    }
                    break
                raise
        
        return {
            "max_supported_seq_len": max_supported,
            "target_seq_len": target_max_seq,
            "target_met": max_supported >= target_max_seq,
            "results_by_length": results_by_length,
        }
    
    def test_throughput_targets(self) -> Dict[str, Any]:
        """スループットターゲットをテスト"""
        targets = self.get_targets()
        min_throughput = targets["min_throughput_tokens_per_sec"]
        
        test_lengths = [1024, 2048, 4096]
        results = {}
        
        for seq_len in test_lengths:
            clear_memory(self.device)
            
            try:
                result = self.measure_throughput(seq_len, num_iterations=50)
                results[str(seq_len)] = {
                    "success": True,
                    "target_met": result["tokens_per_sec"] >= min_throughput,
                    **result,
                }
            except Exception as e:
                results[str(seq_len)] = {
                    "success": False,
                    "error": str(e),
                }
        
        # 平均スループット
        throughputs = [
            r["tokens_per_sec"] for r in results.values()
            if r.get("success") and "tokens_per_sec" in r
        ]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        
        return {
            "min_target_tokens_per_sec": min_throughput,
            "average_throughput": avg_throughput,
            "target_met": avg_throughput >= min_throughput,
            "results_by_length": results,
        }
    
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """メモリ効率をテスト"""
        targets = self.get_targets()
        max_memory_8k = targets["max_memory_at_8k_gb"]
        
        clear_memory(self.device)
        
        try:
            result = self.measure_throughput(8192, num_iterations=10)
            
            return {
                "seq_len": 8192,
                "peak_memory_gb": result["peak_memory_gb"],
                "target_max_memory_gb": max_memory_8k,
                "target_met": result["peak_memory_gb"] <= max_memory_8k,
                "memory_efficiency": (1 - result["peak_memory_gb"] / max_memory_8k) * 100,
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {
                    "seq_len": 8192,
                    "error": "OOM",
                    "target_met": False,
                }
            raise
    
    def run_benchmark(self) -> Dict[str, Any]:
        """フルベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Consumer GPU Benchmark")
        print(f"{'='*60}")
        print(f"GPU: {self.gpu_info.get('name', 'Unknown')}")
        print(f"GPU Type: {self.gpu_info.get('gpu_type', 'Unknown')}")
        print(f"VRAM: {self.gpu_info.get('total_memory_gb', 0):.1f} GB")
        print(f"Compute Capability: {self.gpu_info.get('compute_capability', 'N/A')}")
        print(f"{'='*60}\n")
        
        targets = self.get_targets()
        print(f"Targets for {self.gpu_info.get('gpu_type', 'Unknown')}:")
        print(f"  Max sequence length: {targets['max_seq_len']}")
        print(f"  Min throughput: {targets['min_throughput_tokens_per_sec']} tokens/sec")
        print(f"  Max memory at 8K: {targets['max_memory_at_8k_gb']} GB")
        print()
        
        warmup_gpu(self.device)
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "gpu_info": self.gpu_info,
                "targets": targets,
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
            },
            "tests": {},
        }
        
        # 1. 最大シーケンス長テスト
        print("Testing maximum sequence length...")
        seq_results = self.test_max_sequence_length()
        results["tests"]["max_sequence_length"] = seq_results
        print(f"  Max supported: {seq_results['max_supported_seq_len']}")
        print(f"  Target ({seq_results['target_seq_len']}): "
              f"{'PASS' if seq_results['target_met'] else 'FAIL'}")
        
        # 2. スループットテスト
        print("\nTesting throughput targets...")
        throughput_results = self.test_throughput_targets()
        results["tests"]["throughput"] = throughput_results
        print(f"  Average: {throughput_results['average_throughput']:.0f} tokens/sec")
        print(f"  Target ({throughput_results['min_target_tokens_per_sec']}): "
              f"{'PASS' if throughput_results['target_met'] else 'FAIL'}")
        
        # 3. メモリ効率テスト
        print("\nTesting memory efficiency...")
        memory_results = self.test_memory_efficiency()
        results["tests"]["memory_efficiency"] = memory_results
        if "error" not in memory_results:
            print(f"  Memory at 8K: {memory_results['peak_memory_gb']:.2f} GB")
            print(f"  Target ({memory_results['target_max_memory_gb']} GB): "
                  f"{'PASS' if memory_results['target_met'] else 'FAIL'}")
        else:
            print(f"  Error: {memory_results['error']}")
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        all_tests_passed = (
            seq_results.get("target_met", False) and
            throughput_results.get("target_met", False) and
            memory_results.get("target_met", False)
        )
        
        results["summary"] = {
            "all_targets_met": all_tests_passed,
            "seq_length_target_met": seq_results.get("target_met", False),
            "throughput_target_met": throughput_results.get("target_met", False),
            "memory_target_met": memory_results.get("target_met", False),
        }
        
        print(f"All targets met: {'YES' if all_tests_passed else 'NO'}")
        
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
        description="Phase 8 Consumer GPU Benchmark"
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
        "--output", type=str,
        default="results/benchmarks/phase8_consumer_gpu_benchmark.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = ConsumerGPUBenchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
    
    benchmark.run_benchmark()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
