"""
Phase 2 Benchmark Test Suite

このモジュールは、Phase 2モデルの性能ベンチマークテストを提供します。

テスト内容:
1. BK-Core Tritonカーネルのベンチマーク
   - PyTorch実装との速度比較
   - 数値精度検証
   - 複数のシーケンス長での性能測定

2. メモリ使用量のベンチマーク
   - VRAM使用量の測定
   - バッチサイズとシーケンス長の影響
   - メモリ効率の検証

3. スループットのベンチマーク
   - Forward/Backward速度の測定
   - トークン処理速度の計算
   - 学習時のスループット測定

KPI:
- BK-Core Triton: 3.0倍以上の高速化
- VRAM: 8.0GB未満 (Batch=1, Seq=4096, fp16)
- スループット: 100 tokens/sec以上

Requirements: 11.2
Author: Project MUSE Team
Date: 2025-01-20
"""

import pytest
import torch
import time
import json
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import warnings

# Phase 2モジュール
from src.models.phase2.integrated_model import Phase2IntegratedModel
from src.models.phase2.factory import create_phase2_model, Phase2Config
from src.models.bk_core import BKCoreFunction, set_triton_mode, get_triton_mode

# テスト設定
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        from src.kernels.bk_scan import is_triton_available as check_triton
        return check_triton()
    except Exception:
        return False



# ============================================================================
# BK-Core Tritonカーネルのベンチマーク
# ============================================================================

class TestBKCoreTritonBenchmark:
    """
    BK-Core Tritonカーネルのベンチマーク
    
    PyTorch実装との速度比較と数値精度検証を行います。
    
    KPI: 3.0倍以上の高速化、MSE誤差 1e-6以下
    Requirements: 11.2
    """
    
    def benchmark_bk_core(
        self,
        batch_size: int,
        seq_len: int,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        BK-Coreのベンチマークを実行
        
        Args:
            batch_size: バッチサイズ
            seq_len: シーケンス長
            num_runs: ベンチマーク実行回数
            warmup_runs: ウォームアップ実行回数
        
        Returns:
            results: ベンチマーク結果の辞書
        """
        device = TEST_DEVICE
        
        # テストデータ生成
        torch.manual_seed(42)
        he_diag = torch.randn(batch_size, seq_len, device=device)
        h0_super = torch.randn(batch_size, seq_len - 1, device=device)
        h0_sub = torch.randn(batch_size, seq_len - 1, device=device)
        z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
        
        results = {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
                "device": str(device),
            },
            "pytorch": {},
            "triton": {},
            "speedup": 0.0,
            "numerical_error": 0.0,
        }
        
        # PyTorch実装のベンチマーク
        set_triton_mode(False)
        
        # ウォームアップ
        for _ in range(warmup_runs):
            _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
            if device.type == "cuda":
                torch.cuda.synchronize()
        
        # ベンチマーク
        times_pytorch = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times_pytorch.append((end - start) * 1000)  # ms
        
        pytorch_mean = sum(times_pytorch) / len(times_pytorch)
        pytorch_std = (sum((t - pytorch_mean) ** 2 for t in times_pytorch) / len(times_pytorch)) ** 0.5
        
        results["pytorch"] = {
            "mean_ms": pytorch_mean,
            "std_ms": pytorch_std,
            "min_ms": min(times_pytorch),
            "max_ms": max(times_pytorch),
        }
        
        # Triton実装のベンチマーク
        if not is_triton_available():
            results["triton"]["available"] = False
            return results
        
        set_triton_mode(True)
        results["triton"]["available"] = True
        
        # ウォームアップ
        for _ in range(warmup_runs):
            _ = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
            if device.type == "cuda":
                torch.cuda.synchronize()
        
        # ベンチマーク
        times_triton = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times_triton.append((end - start) * 1000)  # ms
        
        triton_mean = sum(times_triton) / len(times_triton)
        triton_std = (sum((t - triton_mean) ** 2 for t in times_triton) / len(times_triton)) ** 0.5
        
        results["triton"] = {
            "available": True,
            "mean_ms": triton_mean,
            "std_ms": triton_std,
            "min_ms": min(times_triton),
            "max_ms": max(times_triton),
        }
        
        # 高速化率を計算
        speedup = pytorch_mean / triton_mean
        results["speedup"] = speedup
        
        # 数値誤差を計算
        diff = output_pytorch - output_triton
        mse = (diff ** 2).mean().item()
        results["numerical_error"] = mse
        
        return results

    
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_bk_core_small_sequence(self):
        """小規模シーケンスでのベンチマーク (N=512)"""
        results = self.benchmark_bk_core(batch_size=16, seq_len=512, num_runs=50)
        
        print(f"\nBK-Core Benchmark (Small Sequence):")
        print(f"  PyTorch: {results['pytorch']['mean_ms']:.3f} ± {results['pytorch']['std_ms']:.3f} ms")
        print(f"  Triton:  {results['triton']['mean_ms']:.3f} ± {results['triton']['std_ms']:.3f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  MSE:     {results['numerical_error']:.2e}")
        
        # KPI検証
        assert results['speedup'] >= 3.0, f"Speedup {results['speedup']:.2f}x < 3.0x"
        assert results['numerical_error'] < 1e-6, f"MSE {results['numerical_error']:.2e} >= 1e-6"
    
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_bk_core_medium_sequence(self):
        """中規模シーケンスでのベンチマーク (N=2048)"""
        results = self.benchmark_bk_core(batch_size=16, seq_len=2048, num_runs=50)
        
        print(f"\nBK-Core Benchmark (Medium Sequence):")
        print(f"  PyTorch: {results['pytorch']['mean_ms']:.3f} ± {results['pytorch']['std_ms']:.3f} ms")
        print(f"  Triton:  {results['triton']['mean_ms']:.3f} ± {results['triton']['std_ms']:.3f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  MSE:     {results['numerical_error']:.2e}")
        
        # KPI検証
        assert results['speedup'] >= 3.0, f"Speedup {results['speedup']:.2f}x < 3.0x"
        assert results['numerical_error'] < 1e-6, f"MSE {results['numerical_error']:.2e} >= 1e-6"
    
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_bk_core_large_sequence(self):
        """大規模シーケンスでのベンチマーク (N=4096) - KPI測定条件"""
        results = self.benchmark_bk_core(batch_size=16, seq_len=4096, num_runs=100)
        
        print(f"\nBK-Core Benchmark (Large Sequence - KPI):")
        print(f"  PyTorch: {results['pytorch']['mean_ms']:.3f} ± {results['pytorch']['std_ms']:.3f} ms")
        print(f"  Triton:  {results['triton']['mean_ms']:.3f} ± {results['triton']['std_ms']:.3f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  MSE:     {results['numerical_error']:.2e}")
        
        # KPI検証
        assert results['speedup'] >= 3.0, f"Speedup {results['speedup']:.2f}x < 3.0x (KPI FAILED)"
        assert results['numerical_error'] < 1e-6, f"MSE {results['numerical_error']:.2e} >= 1e-6 (KPI FAILED)"
        
        # 結果を保存
        results["kpi_status"] = "PASSED"
        results["timestamp"] = datetime.now().isoformat()
        
        output_file = RESULTS_DIR / "bk_core_triton_benchmark_kpi.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results saved to: {output_file}")
        print(f"  KPI STATUS: PASSED ✓")
    
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_bk_core_scaling(self):
        """スケーリング特性のベンチマーク"""
        sequence_lengths = [256, 512, 1024, 2048, 4096]
        batch_size = 8
        
        scaling_results = {
            "batch_size": batch_size,
            "sequence_lengths": sequence_lengths,
            "pytorch_times": [],
            "triton_times": [],
            "speedups": [],
            "numerical_errors": [],
        }
        
        print(f"\nBK-Core Scaling Benchmark:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence lengths: {sequence_lengths}")
        print()
        
        for seq_len in sequence_lengths:
            results = self.benchmark_bk_core(
                batch_size=batch_size,
                seq_len=seq_len,
                num_runs=30
            )
            
            scaling_results["pytorch_times"].append(results["pytorch"]["mean_ms"])
            scaling_results["triton_times"].append(results["triton"]["mean_ms"])
            scaling_results["speedups"].append(results["speedup"])
            scaling_results["numerical_errors"].append(results["numerical_error"])
            
            print(f"  N={seq_len:4d}: PyTorch={results['pytorch']['mean_ms']:6.2f}ms, "
                  f"Triton={results['triton']['mean_ms']:6.2f}ms, "
                  f"Speedup={results['speedup']:.2f}x, "
                  f"MSE={results['numerical_error']:.2e}")
        
        # すべてのシーケンス長で3倍以上の高速化を達成していることを確認
        min_speedup = min(scaling_results["speedups"])
        assert min_speedup >= 3.0, f"Minimum speedup {min_speedup:.2f}x < 3.0x"
        
        # 結果を保存
        scaling_results["timestamp"] = datetime.now().isoformat()
        output_file = RESULTS_DIR / "bk_core_triton_scaling.json"
        with open(output_file, 'w') as f:
            json.dump(scaling_results, f, indent=2)
        
        print(f"\n  Scaling results saved to: {output_file}")
        print(f"  Minimum speedup: {min_speedup:.2f}x ✓")


# ============================================================================
# メモリ使用量のベンチマーク
# ============================================================================

class TestMemoryBenchmark:
    """
    メモリ使用量のベンチマーク
    
    VRAM使用量を測定し、8GB制約を満たすことを検証します。
    
    KPI: 8.0GB未満 (Batch=1, Seq=4096, fp16)
    Requirements: 11.2
    """
    
    def measure_memory(
        self,
        model: torch.nn.Module,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[str, Any]:
        """
        メモリ使用量を測定
        
        Args:
            model: テスト対象のモデル
            batch_size: バッチサイズ
            seq_len: シーケンス長
            dtype: データ型
        
        Returns:
            results: メモリ使用量の測定結果
        """
        device = TEST_DEVICE
        
        if device.type != "cuda":
            return {
                "device": str(device),
                "error": "CUDA not available",
            }
        
        # モデルをdtypeに変換
        if dtype == torch.float16:
            model = model.half()
        elif dtype == torch.float32:
            model = model.float()
        
        model.eval()
        
        # メモリをリセット
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # ダミー入力
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        # メモリ使用量を測定
        memory_bytes = torch.cuda.max_memory_allocated()
        memory_mb = memory_bytes / (1024 ** 2)
        memory_gb = memory_bytes / (1024 ** 3)
        
        results = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dtype": str(dtype),
            "memory_bytes": memory_bytes,
            "memory_mb": memory_mb,
            "memory_gb": memory_gb,
            "device": str(device),
        }
        
        return results

    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_small_model(self):
        """小規模モデルのメモリ使用量"""
        model = create_phase2_model(preset="small", device=TEST_DEVICE)
        
        results = self.measure_memory(model, batch_size=4, seq_len=1024, dtype=torch.float16)
        
        print(f"\nMemory Benchmark (Small Model):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Data type: {results['dtype']}")
        print(f"  Memory usage: {results['memory_gb']:.2f} GB ({results['memory_mb']:.1f} MB)")
        
        # 小規模モデルは2GB以下であるべき
        assert results['memory_gb'] < 2.0, f"Memory {results['memory_gb']:.2f} GB >= 2.0 GB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_base_model(self):
        """標準モデルのメモリ使用量"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        
        results = self.measure_memory(model, batch_size=2, seq_len=2048, dtype=torch.float16)
        
        print(f"\nMemory Benchmark (Base Model):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Data type: {results['dtype']}")
        print(f"  Memory usage: {results['memory_gb']:.2f} GB ({results['memory_mb']:.1f} MB)")
        
        # 標準モデルは5GB以下であるべき
        assert results['memory_gb'] < 5.0, f"Memory {results['memory_gb']:.2f} GB >= 5.0 GB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_kpi(self):
        """KPI条件でのメモリ使用量 (Batch=1, Seq=4096, fp16)"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        
        results = self.measure_memory(model, batch_size=1, seq_len=4096, dtype=torch.float16)
        
        print(f"\nMemory Benchmark (KPI Condition):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Data type: {results['dtype']}")
        print(f"  Memory usage: {results['memory_gb']:.2f} GB ({results['memory_mb']:.1f} MB)")
        
        # KPI検証: 8.0GB未満
        target_memory_gb = 8.0
        passed = results['memory_gb'] < target_memory_gb
        
        results["target_memory_gb"] = target_memory_gb
        results["kpi_status"] = "PASSED" if passed else "FAILED"
        results["timestamp"] = datetime.now().isoformat()
        
        # 結果を保存
        output_file = RESULTS_DIR / "phase2_memory_kpi.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Target: < {target_memory_gb} GB")
        print(f"  Results saved to: {output_file}")
        print(f"  KPI STATUS: {results['kpi_status']} {'✓' if passed else '✗'}")
        
        assert passed, f"Memory {results['memory_gb']:.2f} GB >= {target_memory_gb} GB (KPI FAILED)"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_scaling(self):
        """バッチサイズとシーケンス長によるメモリスケーリング"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        
        test_configs = [
            (1, 512),
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (2, 512),
            (4, 512),
            (8, 512),
        ]
        
        scaling_results = {
            "configs": [],
            "memory_gb": [],
        }
        
        print(f"\nMemory Scaling Benchmark:")
        print(f"  Model: base")
        print(f"  Data type: fp16")
        print()
        
        for batch_size, seq_len in test_configs:
            results = self.measure_memory(model, batch_size, seq_len, dtype=torch.float16)
            
            scaling_results["configs"].append(f"B={batch_size}, N={seq_len}")
            scaling_results["memory_gb"].append(results["memory_gb"])
            
            print(f"  B={batch_size:2d}, N={seq_len:4d}: {results['memory_gb']:.2f} GB")
        
        # 結果を保存
        scaling_results["timestamp"] = datetime.now().isoformat()
        output_file = RESULTS_DIR / "phase2_memory_scaling.json"
        with open(output_file, 'w') as f:
            json.dump(scaling_results, f, indent=2)
        
        print(f"\n  Scaling results saved to: {output_file}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_dtype_comparison(self):
        """データ型によるメモリ使用量の比較"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        
        batch_size = 1
        seq_len = 2048
        
        print(f"\nMemory Data Type Comparison:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print()
        
        dtype_results = {}
        
        for dtype in [torch.float32, torch.float16]:
            results = self.measure_memory(model, batch_size, seq_len, dtype=dtype)
            dtype_results[str(dtype)] = results["memory_gb"]
            
            print(f"  {str(dtype):20s}: {results['memory_gb']:.2f} GB")
        
        # fp16がfp32の約半分のメモリを使用することを確認
        if torch.float32 in [torch.float32, torch.float16] and torch.float16 in [torch.float32, torch.float16]:
            ratio = dtype_results[str(torch.float32)] / dtype_results[str(torch.float16)]
            print(f"\n  fp32/fp16 ratio: {ratio:.2f}x")
            
            # 比率が1.5~2.5の範囲であることを確認（完全に2倍にはならない）
            assert 1.5 < ratio < 2.5, f"Unexpected memory ratio: {ratio:.2f}x"


# ============================================================================
# スループットのベンチマーク
# ============================================================================

class TestThroughputBenchmark:
    """
    スループットのベンチマーク
    
    Forward/Backward速度とトークン処理速度を測定します。
    
    KPI: 100 tokens/sec以上
    Requirements: 11.2
    """
    
    def measure_throughput(
        self,
        model: torch.nn.Module,
        batch_size: int,
        seq_len: int,
        num_iterations: int = 10,
        measure_backward: bool = False,
    ) -> Dict[str, Any]:
        """
        スループットを測定
        
        Args:
            model: テスト対象のモデル
            batch_size: バッチサイズ
            seq_len: シーケンス長
            num_iterations: 測定回数
            measure_backward: Backwardも測定するか
        
        Returns:
            results: スループット測定結果
        """
        device = TEST_DEVICE
        model.eval()
        
        # ウォームアップ
        warmup_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            _ = model(warmup_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Forward pass測定
        total_tokens = 0
        forward_times = []
        
        for _ in range(num_iterations):
            input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                logits = model(input_ids)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            forward_times.append(end - start)
            total_tokens += batch_size * seq_len
        
        forward_time = sum(forward_times)
        forward_throughput = total_tokens / forward_time
        
        results = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_iterations": num_iterations,
            "total_tokens": total_tokens,
            "forward_time": forward_time,
            "forward_throughput": forward_throughput,
            "device": str(device),
        }
        
        # Backward pass測定（オプション）
        if measure_backward:
            model.train()
            backward_times = []
            
            for _ in range(num_iterations):
                input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                logits = model(input_ids)
                loss = logits.sum()
                loss.backward()
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                backward_times.append(end - start)
                
                # 勾配をクリア
                model.zero_grad()
            
            backward_time = sum(backward_times)
            backward_throughput = total_tokens / backward_time
            
            results["backward_time"] = backward_time
            results["backward_throughput"] = backward_throughput
            
            model.eval()
        
        return results

    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput_small_model(self):
        """小規模モデルのスループット"""
        model = create_phase2_model(preset="small", device=TEST_DEVICE)
        model = model.half()  # fp16
        
        results = self.measure_throughput(model, batch_size=4, seq_len=512, num_iterations=20)
        
        print(f"\nThroughput Benchmark (Small Model):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Forward time: {results['forward_time']:.2f} sec")
        print(f"  Forward throughput: {results['forward_throughput']:.1f} tokens/sec")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput_base_model(self):
        """標準モデルのスループット"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        model = model.half()  # fp16
        
        results = self.measure_throughput(model, batch_size=4, seq_len=512, num_iterations=20)
        
        print(f"\nThroughput Benchmark (Base Model):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Forward time: {results['forward_time']:.2f} sec")
        print(f"  Forward throughput: {results['forward_throughput']:.1f} tokens/sec")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput_kpi(self):
        """KPI条件でのスループット測定"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        model = model.half()  # fp16
        
        results = self.measure_throughput(model, batch_size=4, seq_len=512, num_iterations=20)
        
        print(f"\nThroughput Benchmark (KPI):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Forward time: {results['forward_time']:.2f} sec")
        print(f"  Forward throughput: {results['forward_throughput']:.1f} tokens/sec")
        
        # KPI検証: 100 tokens/sec以上
        target_throughput = 100.0
        passed = results['forward_throughput'] >= target_throughput
        
        results["target_throughput"] = target_throughput
        results["kpi_status"] = "PASSED" if passed else "FAILED"
        results["timestamp"] = datetime.now().isoformat()
        
        # 結果を保存
        output_file = RESULTS_DIR / "phase2_throughput_kpi.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Target: >= {target_throughput} tokens/sec")
        print(f"  Results saved to: {output_file}")
        print(f"  KPI STATUS: {results['kpi_status']} {'✓' if passed else '✗'}")
        
        # 注: スループットはハードウェアに依存するため、警告のみ
        if not passed:
            warnings.warn(
                f"Throughput {results['forward_throughput']:.1f} tokens/sec is below target {target_throughput} tokens/sec. "
                f"This may be due to hardware limitations.",
                UserWarning
            )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput_with_backward(self):
        """Forward + Backwardのスループット"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        model = model.half()  # fp16
        
        results = self.measure_throughput(
            model,
            batch_size=4,
            seq_len=512,
            num_iterations=10,
            measure_backward=True
        )
        
        print(f"\nThroughput Benchmark (Forward + Backward):")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Sequence length: {results['seq_len']}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Forward time: {results['forward_time']:.2f} sec")
        print(f"  Forward throughput: {results['forward_throughput']:.1f} tokens/sec")
        print(f"  Backward time: {results['backward_time']:.2f} sec")
        print(f"  Backward throughput: {results['backward_throughput']:.1f} tokens/sec")
        
        # Backwardの方が遅いことを確認
        assert results['backward_time'] > results['forward_time'], \
            "Backward should be slower than forward"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput_scaling(self):
        """バッチサイズによるスループットのスケーリング"""
        model = create_phase2_model(preset="base", device=TEST_DEVICE)
        model = model.half()  # fp16
        
        batch_sizes = [1, 2, 4, 8]
        seq_len = 512
        
        scaling_results = {
            "seq_len": seq_len,
            "batch_sizes": batch_sizes,
            "throughputs": [],
        }
        
        print(f"\nThroughput Scaling Benchmark:")
        print(f"  Sequence length: {seq_len}")
        print()
        
        for batch_size in batch_sizes:
            results = self.measure_throughput(model, batch_size, seq_len, num_iterations=10)
            
            scaling_results["throughputs"].append(results["forward_throughput"])
            
            print(f"  Batch={batch_size}: {results['forward_throughput']:.1f} tokens/sec")
        
        # バッチサイズが大きいほどスループットが高いことを確認
        # （並列処理の効率が上がるため）
        assert scaling_results["throughputs"][-1] > scaling_results["throughputs"][0], \
            "Throughput should increase with batch size"
        
        # 結果を保存
        scaling_results["timestamp"] = datetime.now().isoformat()
        output_file = RESULTS_DIR / "phase2_throughput_scaling.json"
        with open(output_file, 'w') as f:
            json.dump(scaling_results, f, indent=2)
        
        print(f"\n  Scaling results saved to: {output_file}")


# ============================================================================
# 総合ベンチマークレポート生成
# ============================================================================

class TestBenchmarkReport:
    """
    総合ベンチマークレポートの生成
    
    すべてのベンチマーク結果をまとめたレポートを生成します。
    
    Requirements: 11.2
    """
    
    def test_generate_comprehensive_report(self):
        """総合ベンチマークレポートの生成"""
        print(f"\n{'='*80}")
        print("Phase 2 Comprehensive Benchmark Report")
        print(f"{'='*80}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Device: {TEST_DEVICE}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        print(f"{'='*80}")
        print()
        
        # レポートデータを収集
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.system(),
                "device": str(TEST_DEVICE),
                "cuda_available": torch.cuda.is_available(),
            },
            "kpi_summary": {},
            "benchmark_files": [],
        }
        
        if torch.cuda.is_available():
            report["metadata"]["gpu_name"] = torch.cuda.get_device_name(0)
            report["metadata"]["cuda_version"] = torch.version.cuda
        
        # 既存のベンチマーク結果ファイルを読み込み
        benchmark_files = [
            "bk_core_triton_benchmark_kpi.json",
            "phase2_memory_kpi.json",
            "phase2_throughput_kpi.json",
        ]
        
        print("KPI Summary:")
        print("-" * 80)
        
        for filename in benchmark_files:
            filepath = RESULTS_DIR / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                report["benchmark_files"].append(filename)
                
                # KPIステータスを抽出
                if "kpi_status" in data:
                    kpi_name = filename.replace(".json", "").replace("_", " ").title()
                    status = data["kpi_status"]
                    report["kpi_summary"][kpi_name] = status
                    
                    print(f"  {kpi_name:40s}: {status:10s} {'✓' if status == 'PASSED' else '✗'}")
        
        print("-" * 80)
        print()
        
        # レポートを保存
        report_file = RESULTS_DIR / "phase2_benchmark_comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive report saved to: {report_file}")
        print()
        
        # Markdownレポートも生成
        self._generate_markdown_report(report)
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Markdownフォーマットのレポートを生成"""
        md_lines = [
            "# Phase 2 Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Platform:** {report['metadata']['platform']}",
            f"**Device:** {report['metadata']['device']}",
        ]
        
        if report['metadata']['cuda_available']:
            md_lines.append(f"**GPU:** {report['metadata'].get('gpu_name', 'Unknown')}")
            md_lines.append(f"**CUDA Version:** {report['metadata'].get('cuda_version', 'Unknown')}")
        
        md_lines.extend([
            "",
            "## KPI Summary",
            "",
            "| KPI | Status |",
            "|-----|--------|",
        ])
        
        for kpi_name, status in report['kpi_summary'].items():
            status_icon = "✓" if status == "PASSED" else "✗"
            md_lines.append(f"| {kpi_name} | {status} {status_icon} |")
        
        md_lines.extend([
            "",
            "## Benchmark Files",
            "",
        ])
        
        for filename in report['benchmark_files']:
            md_lines.append(f"- `{filename}`")
        
        md_lines.extend([
            "",
            "## Notes",
            "",
            "- BK-Core Triton: Target 3.0x+ speedup",
            "- Memory: Target < 8.0 GB (Batch=1, Seq=4096, fp16)",
            "- Throughput: Target >= 100 tokens/sec",
            "",
            "For detailed results, see individual JSON files in `results/benchmarks/`.",
            "",
        ])
        
        # Markdownファイルを保存
        md_file = RESULTS_DIR / "PHASE2_BENCHMARK_REPORT.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        print(f"Markdown report saved to: {md_file}")


# ============================================================================
# テスト実行時の設定
# ============================================================================

if __name__ == "__main__":
    # pytestを使用してテストを実行
    pytest.main([__file__, "-v", "-s"])
