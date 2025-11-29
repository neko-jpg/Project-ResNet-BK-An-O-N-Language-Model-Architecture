"""
Benchmark script for Tasks 9-13 (Phase 8)

タスク9-13の実装検証ベンチマーク:
- Task 10: Tangent-Space Linear Attention
- Task 11: Hybrid Precision Strategy
- Task 13: Block-wise Distance Computation

Requirements: 5.1-5.6, 6.1-6.6, 7.1-7.6, 70.1-70.6
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase8.linear_attention import (
    TangentSpaceLinearAttention,
    LinearAttentionConfig,
    create_linear_attention,
)
from src.models.phase8.precision_manager import (
    HybridPrecisionManager,
    PrecisionConfig,
    BoundaryCollapseGuard,
    create_precision_manager,
)
from src.models.phase8.block_distance import (
    BlockWiseDistanceComputation,
    BlockDistanceConfig,
    create_block_distance,
)


def benchmark_linear_attention():
    """Linear Attentionのベンチマーク"""
    print("=" * 60)
    print("Task 10: Tangent-Space Linear Attention Benchmark")
    print("=" * 60)
    
    results = {
        "task": "10",
        "name": "Tangent-Space Linear Attention",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }
    
    # 複雑度テスト
    print("\n1. Complexity Test (O(N) scaling)")
    config = LinearAttentionConfig(
        d_model=64,
        num_heads=4,
        curvature=0.05,  # 線形モードを強制
    )
    module = TangentSpaceLinearAttention(config)
    module.eval()
    
    seq_lengths = [128, 256, 512, 1024]
    times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 64) * 0.5
        
        # ウォームアップ
        with torch.no_grad():
            _ = module(x)
        
        # 計測
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = module(x)
        elapsed = (time.time() - start) / 10 * 1000  # ms
        times.append(elapsed)
        print(f"  seq_len={seq_len}: {elapsed:.2f} ms")
    
    # スケーリング比率
    scaling_ratios = []
    for i in range(1, len(times)):
        ratio = times[i] / times[i-1]
        seq_ratio = seq_lengths[i] / seq_lengths[i-1]
        scaling_ratios.append(ratio / seq_ratio)
    
    avg_scaling = sum(scaling_ratios) / len(scaling_ratios)
    is_linear = avg_scaling < 1.5  # O(N)なら1.0に近い
    
    results["tests"].append({
        "name": "complexity_scaling",
        "seq_lengths": seq_lengths,
        "times_ms": times,
        "scaling_ratios": scaling_ratios,
        "avg_scaling": avg_scaling,
        "is_linear": is_linear,
        "passed": is_linear,
    })
    print(f"  Average scaling ratio: {avg_scaling:.2f} (target: < 1.5)")
    print(f"  Result: {'PASS' if is_linear else 'FAIL'}")
    
    # モード切替テスト
    print("\n2. Mode Switching Test")
    mode_tests = []
    for curvature, expected_mode in [(0.05, "linear"), (0.5, "hybrid"), (1.5, "exact")]:
        config = LinearAttentionConfig(d_model=64, num_heads=4, curvature=curvature)
        module = TangentSpaceLinearAttention(config)
        
        x = torch.randn(1, 32, 64) * 0.5
        _, diag = module(x, return_diagnostics=True)
        
        passed = diag.mode == expected_mode
        mode_tests.append({
            "curvature": curvature,
            "expected_mode": expected_mode,
            "actual_mode": diag.mode,
            "passed": passed,
        })
        print(f"  curvature={curvature}: mode={diag.mode} (expected: {expected_mode}) - {'PASS' if passed else 'FAIL'}")
    
    results["tests"].append({
        "name": "mode_switching",
        "tests": mode_tests,
        "passed": all(t["passed"] for t in mode_tests),
    })
    
    # 相関テスト
    print("\n3. Correlation with Exact Test")
    config = LinearAttentionConfig(d_model=64, num_heads=4, curvature=0.01)
    module = TangentSpaceLinearAttention(config)
    
    x = torch.randn(1, 16, 64) * 0.3
    correlation = module.compute_correlation_with_exact(x)
    
    passed = correlation > 0.7
    results["tests"].append({
        "name": "correlation_with_exact",
        "correlation": correlation,
        "threshold": 0.7,
        "passed": passed,
    })
    print(f"  Correlation: {correlation:.4f} (threshold: > 0.7)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    results["overall_passed"] = all(t["passed"] for t in results["tests"])
    return results


def benchmark_precision_manager():
    """Precision Managerのベンチマーク"""
    print("\n" + "=" * 60)
    print("Task 11: Hybrid Precision Strategy Benchmark")
    print("=" * 60)
    
    results = {
        "task": "11",
        "name": "Hybrid Precision Strategy",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }
    
    # 曲率精度テスト
    print("\n1. Curvature Precision Enforcement Test")
    config = PrecisionConfig(default_dtype="float16")
    manager = HybridPrecisionManager(config)
    
    def curvature_func(x):
        return x.pow(2).sum(dim=-1)
    
    x_fp16 = torch.randn(2, 16, 64, dtype=torch.float16)
    result = manager.compute_curvature_safe(curvature_func, x_fp16)
    
    passed = result.dtype == torch.float32
    results["tests"].append({
        "name": "curvature_precision",
        "input_dtype": "float16",
        "output_dtype": str(result.dtype),
        "expected_dtype": "float32",
        "passed": passed,
    })
    print(f"  Input dtype: float16, Output dtype: {result.dtype}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # 境界検出テスト
    print("\n2. Boundary Detection Test")
    x_boundary = torch.zeros(1, 10, 64)
    x_boundary[0, 0, 0] = 0.96  # 境界近く
    x_boundary[0, 1, 0] = 0.97
    
    count = manager.boundary_detector.count_boundary_tokens(x_boundary)
    passed = count == 2
    
    results["tests"].append({
        "name": "boundary_detection",
        "boundary_tokens_count": count,
        "expected_count": 2,
        "passed": passed,
    })
    print(f"  Boundary tokens detected: {count} (expected: 2)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # 境界崩壊防止テスト
    print("\n3. Boundary Collapse Prevention Test")
    guard = BoundaryCollapseGuard(max_norm=0.99)
    
    x_large = torch.zeros(1, 10, 64)
    x_large[0, 0, :] = 10.0  # 非常に大きなノルム
    
    x_safe = guard(x_large)
    norms = x_safe.norm(dim=-1)
    
    passed = (norms <= 0.99 + 1e-6).all().item()
    results["tests"].append({
        "name": "boundary_collapse_prevention",
        "max_norm_after": norms.max().item(),
        "threshold": 0.99,
        "passed": passed,
    })
    print(f"  Max norm after guard: {norms.max().item():.4f} (threshold: 0.99)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # オーバーフロー回復テスト
    print("\n4. Overflow Recovery Test")
    x_overflow = torch.tensor([1e10])
    result, recovered = manager.detect_and_recover_overflow(x_overflow)
    
    passed = recovered and result.dtype == torch.float32
    results["tests"].append({
        "name": "overflow_recovery",
        "recovered": recovered,
        "output_dtype": str(result.dtype),
        "passed": passed,
    })
    print(f"  Overflow recovered: {recovered}, Output dtype: {result.dtype}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    results["overall_passed"] = all(t["passed"] for t in results["tests"])
    return results


def benchmark_block_distance():
    """Block-wise Distanceのベンチマーク"""
    print("\n" + "=" * 60)
    print("Task 13: Block-wise Distance Computation Benchmark")
    print("=" * 60)
    
    results = {
        "task": "13",
        "name": "Block-wise Distance Computation",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }
    
    # メモリスケーリングテスト
    print("\n1. Memory Scaling Test (O(N))")
    config = BlockDistanceConfig(
        d_model=64,
        num_heads=4,
        block_size_m=32,
        block_size_n=32,
    )
    module = BlockWiseDistanceComputation(config)
    
    seq_lengths = [128, 256, 512, 1024]
    memories = []
    
    for seq_len in seq_lengths:
        mem = module.estimate_memory_usage(seq_len)
        memories.append(mem)
        print(f"  seq_len={seq_len}: {mem:.2f} MB")
    
    # スケーリング比率
    scaling_ratios = []
    for i in range(1, len(memories)):
        ratio = memories[i] / memories[i-1]
        seq_ratio = seq_lengths[i] / seq_lengths[i-1]
        scaling_ratios.append(ratio / seq_ratio)
    
    avg_scaling = sum(scaling_ratios) / len(scaling_ratios)
    is_linear = avg_scaling < 1.5
    
    results["tests"].append({
        "name": "memory_scaling",
        "seq_lengths": seq_lengths,
        "memories_mb": memories,
        "scaling_ratios": scaling_ratios,
        "avg_scaling": avg_scaling,
        "is_linear": is_linear,
        "passed": is_linear,
    })
    print(f"  Average scaling ratio: {avg_scaling:.2f} (target: < 1.5)")
    print(f"  Result: {'PASS' if is_linear else 'FAIL'}")
    
    # Causalブロックスキップテスト
    print("\n2. Causal Block Skipping Test")
    config = BlockDistanceConfig(
        d_model=64,
        num_heads=4,
        block_size_m=16,
        block_size_n=16,
        causal=True,
    )
    module = BlockWiseDistanceComputation(config)
    
    x = torch.randn(1, 64, 64) * 0.5
    _, diag = module(x, return_diagnostics=True)
    
    # 4x4 = 16ブロック中、上三角6ブロックがスキップ
    expected_skipped = 6
    expected_computed = 10
    
    passed = diag.num_blocks_skipped == expected_skipped and diag.num_blocks_computed == expected_computed
    results["tests"].append({
        "name": "causal_block_skipping",
        "blocks_skipped": diag.num_blocks_skipped,
        "blocks_computed": diag.num_blocks_computed,
        "expected_skipped": expected_skipped,
        "expected_computed": expected_computed,
        "passed": passed,
    })
    print(f"  Blocks skipped: {diag.num_blocks_skipped} (expected: {expected_skipped})")
    print(f"  Blocks computed: {diag.num_blocks_computed} (expected: {expected_computed})")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # 正確性テスト
    print("\n3. Correctness Test")
    config = BlockDistanceConfig(
        d_model=64,
        num_heads=4,
        block_size_m=32,
        block_size_n=32,
    )
    module = BlockWiseDistanceComputation(config)
    
    x = torch.randn(2, 64, 64) * 0.5
    output, _ = module(x)
    
    is_finite = torch.isfinite(output).all().item()
    correct_shape = output.shape == x.shape
    
    passed = is_finite and correct_shape
    results["tests"].append({
        "name": "correctness",
        "output_finite": is_finite,
        "correct_shape": correct_shape,
        "passed": passed,
    })
    print(f"  Output finite: {is_finite}, Correct shape: {correct_shape}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    results["overall_passed"] = all(t["passed"] for t in results["tests"])
    return results


def main():
    """メインベンチマーク実行"""
    print("=" * 60)
    print("Phase 8 Tasks 9-13 Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "tasks": [],
    }
    
    # 各タスクのベンチマーク
    task10_results = benchmark_linear_attention()
    all_results["tasks"].append(task10_results)
    
    task11_results = benchmark_precision_manager()
    all_results["tasks"].append(task11_results)
    
    task13_results = benchmark_block_distance()
    all_results["tasks"].append(task13_results)
    
    # 全体結果
    all_passed = all(t["overall_passed"] for t in all_results["tasks"])
    all_results["overall_passed"] = all_passed
    
    # 結果をJSONに保存
    output_path = Path("results/benchmarks/TASK9_13_BENCHMARK_RESULTS.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task in all_results["tasks"]:
        status = "PASS" if task["overall_passed"] else "FAIL"
        print(f"  Task {task['task']} ({task['name']}): {status}")
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print(f"\nResults saved to: {output_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
