#!/usr/bin/env python3
"""
Phase 8 Memory Profiling Benchmark Script

タスク30.2: Phase 8のメモリプロファイリング
- 各シーケンス長でのピークVRAM測定
- torch.cuda.memory_stats()による詳細分析
- 目標: Phase 7比50-80%メモリ削減

Requirements: 30.2
"""

import json
import gc
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

# Phase 7モジュールのインポート
try:
    from src.models.phase7 import HyperbolicAttention
    PHASE7_AVAILABLE = True
except ImportError:
    PHASE7_AVAILABLE = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory(device: torch.device):
    """メモリをクリア"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    """詳細なメモリ統計を取得"""
    if device.type != "cuda":
        return {"error": "CUDA not available"}
    
    stats = torch.cuda.memory_stats()
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "active_blocks": stats.get("active.all.current", 0),
        "inactive_split_blocks": stats.get("inactive_split.all.current", 0),
    }


def measure_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    include_backward: bool = True,
) -> Dict[str, Any]:
    """
    モデルのメモリ使用量を測定
    
    Args:
        model: 測定対象のモデル
        batch_size: バッチサイズ
        seq_len: シーケンス長
        d_model: モデル次元
        device: デバイス
        include_backward: 逆伝播も測定するか
    
    Returns:
        メモリ使用量の詳細
    """
    model = model.to(device)
    
    # メモリクリア
    clear_memory(device)
    
    # 入力データ生成
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=include_backward)
    
    # Forward pass
    model.train() if include_backward else model.eval()
    
    try:
        if include_backward:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            
            # Forward後のメモリ
            forward_stats = get_memory_stats(device)
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Backward後のメモリ
            backward_stats = get_memory_stats(device)
            
            return {
                "success": True,
                "forward": forward_stats,
                "backward": backward_stats,
                "peak_memory_mb": backward_stats["peak_allocated_mb"],
                "theoretical_o_n_memory_mb": batch_size * seq_len * d_model * 4 / (1024 * 1024),
            }
        else:
            with torch.no_grad():
                output = model(x)
            
            inference_stats = get_memory_stats(device)
            
            return {
                "success": True,
                "inference": inference_stats,
                "peak_memory_mb": inference_stats["peak_allocated_mb"],
                "theoretical_o_n_memory_mb": batch_size * seq_len * d_model * 4 / (1024 * 1024),
            }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "success": False,
                "error": "OOM",
                "message": str(e),
            }
        raise


class Phase8MemoryBenchmark:
    """Phase 8メモリベンチマーク"""
    
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
        seq_lengths: List[int] = [1024, 2048, 4096, 8192, 16384],
        include_backward: bool = True,
    ) -> Dict[str, Any]:
        """ベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Memory Profiling Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"Include backward: {include_backward}")
        print(f"{'='*60}\n")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "include_backward": include_backward,
            },
            "phase8": {},
            "phase7_baseline": {},
            "comparison": {},
            "memory_scaling": {},
        }
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # Phase 8メモリ測定
            try:
                phase8_model = self.create_phase8_model(seq_len)
                phase8_results = measure_memory_usage(
                    phase8_model,
                    self.batch_size,
                    seq_len,
                    self.d_model,
                    self.device,
                    include_backward,
                )
                results["phase8"][str(seq_len)] = phase8_results
                
                if phase8_results["success"]:
                    print(f"  Phase 8: {phase8_results['peak_memory_mb']:.1f} MB peak")
                else:
                    print(f"  Phase 8: {phase8_results['error']}")
            except Exception as e:
                print(f"  Phase 8: Error - {e}")
                results["phase8"][str(seq_len)] = {"error": str(e)}
            
            clear_memory(self.device)
            
            # Phase 7メモリ測定
            try:
                phase7_model = self.create_phase7_model(seq_len)
                phase7_results = measure_memory_usage(
                    phase7_model,
                    self.batch_size,
                    seq_len,
                    self.d_model,
                    self.device,
                    include_backward,
                )
                results["phase7_baseline"][str(seq_len)] = phase7_results
                
                if phase7_results["success"]:
                    print(f"  Phase 7: {phase7_results['peak_memory_mb']:.1f} MB peak")
                else:
                    print(f"  Phase 7: {phase7_results['error']}")
            except Exception as e:
                print(f"  Phase 7: Error - {e}")
                results["phase7_baseline"][str(seq_len)] = {"error": str(e)}
            
            clear_memory(self.device)
            
            # 比較計算
            p8 = results["phase8"].get(str(seq_len), {})
            p7 = results["phase7_baseline"].get(str(seq_len), {})
            
            if p8.get("success") and p7.get("success"):
                p8_mem = p8["peak_memory_mb"]
                p7_mem = p7["peak_memory_mb"]
                reduction = (1 - p8_mem / p7_mem) * 100 if p7_mem > 0 else 0
                
                results["comparison"][str(seq_len)] = {
                    "phase8_mb": p8_mem,
                    "phase7_mb": p7_mem,
                    "reduction_percent": reduction,
                    "target_met": 50 <= reduction <= 80,
                }
                
                print(f"  Reduction: {reduction:.1f}% (target: 50-80%)")
        
        # メモリスケーリング分析
        phase8_mems = []
        seq_lens = []
        for seq_len in seq_lengths:
            p8 = results["phase8"].get(str(seq_len), {})
            if p8.get("success"):
                phase8_mems.append(p8["peak_memory_mb"])
                seq_lens.append(seq_len)
        
        if len(phase8_mems) >= 2:
            # O(N)スケーリングの検証
            # メモリがシーケンス長に線形にスケールするか確認
            ratios = []
            for i in range(1, len(phase8_mems)):
                mem_ratio = phase8_mems[i] / phase8_mems[i-1]
                seq_ratio = seq_lens[i] / seq_lens[i-1]
                ratios.append(mem_ratio / seq_ratio)
            
            avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
            is_linear = 0.8 <= avg_ratio <= 1.2  # 線形に近いか
            
            results["memory_scaling"] = {
                "scaling_ratios": ratios,
                "average_ratio": avg_ratio,
                "is_linear_o_n": is_linear,
            }
            
            print(f"\nMemory Scaling Analysis:")
            print(f"  Average ratio: {avg_ratio:.2f} (1.0 = perfect O(N))")
            print(f"  Is O(N): {is_linear}")
        
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
            "average_reduction": sum(
                v["reduction_percent"] for v in results["comparison"].values()
                if isinstance(v, dict) and "reduction_percent" in v
            ) / total_tests if total_tests > 0 else 0,
        }
        
        print(f"Targets met: {targets_met}/{total_tests}")
        print(f"Average reduction: {results['summary']['average_reduction']:.1f}%")
        
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
        description="Phase 8 Memory Profiling Benchmark"
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
        default=[1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark"
    )
    parser.add_argument(
        "--no-backward", action="store_true",
        help="Skip backward pass measurement"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_memory_profile.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = Phase8MemoryBenchmark(
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
    
    benchmark.run_benchmark(
        seq_lengths=args.seq_lengths,
        include_backward=not args.no_backward,
    )
    
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
