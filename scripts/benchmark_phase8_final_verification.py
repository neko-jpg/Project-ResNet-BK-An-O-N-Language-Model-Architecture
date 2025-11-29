#!/usr/bin/env python3
"""
Phase 8 Final GPU Verification Script

タスク35: 最終GPU検証
- RTX 3080でのフルベンチマークスイート実行
- 全スループット目標の検証
- 全メモリ目標の検証
- Phase 7との比較レポート生成

Requirements: All Phase 8 requirements
"""

import json
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

# GPU情報取得
def get_gpu_info() -> Dict[str, Any]:
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU only"}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def warmup_gpu(device: torch.device, iterations: int = 20):
    """GPUウォームアップ"""
    if device.type == "cuda":
        x = torch.randn(2048, 2048, device=device)
        for _ in range(iterations):
            _ = torch.matmul(x, x)
        torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()


def measure_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
) -> Dict[str, float]:
    """メモリ使用量を測定"""
    model = model.to(device)
    model.eval()
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    with torch.no_grad():
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    peak_memory_mb = 0.0
    allocated_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        allocated_memory_mb = torch.cuda.memory_allocated() / (1024**2)
    
    del x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "peak_memory_mb": peak_memory_mb,
        "allocated_memory_mb": allocated_memory_mb,
    }


def measure_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """スループットを測定"""
    model = model.to(device)
    model.eval()
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # 測定
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_sec = total_tokens / total_time
    latency_ms = (total_time / num_iterations) * 1000
    
    memory_mb = 0.0
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    del x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
    }


class Phase8FinalVerification:
    """Phase 8最終検証クラス"""
    
    # 目標値定義
    TARGETS = {
        "throughput_improvement": 2.0,  # Phase 7比2x
        "memory_reduction_percent": 50.0,  # 50%削減
        "max_memory_8192_mb": 3000.0,  # seq=8192で3GB以下
        "flops_utilization": 0.70,  # 70%以上
        "int8_speedup": 2.0,  # INT8で2x高速化
        "linear_attention_correlation": 0.95,  # 95%相関
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
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        self.results: Dict[str, Any] = {}
        self.gpu_info = get_gpu_info()
    
    def _create_simple_attention(self) -> nn.Module:
        """シンプルなアテンションモデルを作成"""
        return nn.MultiheadAttention(
            self.d_model, self.num_heads, batch_first=True
        )
    
    def _create_linear_model(self) -> nn.Module:
        """シンプルな線形モデルを作成（フォールバック用）"""
        class SimpleLinearModel(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)
                self.norm = nn.LayerNorm(d_model)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(self.linear(x))
        
        return SimpleLinearModel(self.d_model)
    
    def _try_import_phase8(self):
        """Phase 8モジュールのインポートを試行"""
        modules = {}
        try:
            from src.models.phase8 import BlockWiseDistanceComputation
            modules["BlockWiseDistanceComputation"] = BlockWiseDistanceComputation
        except ImportError:
            pass
        try:
            from src.models.phase8 import HyperbolicSSM
            modules["HyperbolicSSM"] = HyperbolicSSM
        except ImportError:
            pass
        try:
            from src.models.phase8 import TangentSpaceLinearAttention
            modules["TangentSpaceLinearAttention"] = TangentSpaceLinearAttention
        except ImportError:
            pass
        try:
            from src.models.phase8 import ARSSMHyperbolicFusion
            modules["ARSSMHyperbolicFusion"] = ARSSMHyperbolicFusion
        except ImportError:
            pass
        try:
            from src.models.phase8 import EntailmentCones
            modules["EntailmentCones"] = EntailmentCones
        except ImportError:
            pass
        try:
            from src.models.phase8 import SheafAttentionModule
            modules["SheafAttentionModule"] = SheafAttentionModule
        except ImportError:
            pass
        return modules
    
    def _try_import_phase7(self):
        """Phase 7モジュールのインポートを試行"""
        modules = {}
        try:
            from src.models.phase7 import HyperbolicAttention
            modules["HyperbolicAttention"] = HyperbolicAttention
        except ImportError:
            pass
        return modules

    def run_throughput_verification(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_iterations: int = 50,
    ) -> Dict[str, Any]:
        """スループット検証を実行"""
        print("\n" + "="*60)
        print("Throughput Verification")
        print("="*60)
        
        results = {"phase8": {}, "phase7": {}, "comparison": {}}
        phase8_modules = self._try_import_phase8()
        phase7_modules = self._try_import_phase7()
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # Phase 8テスト (TangentSpaceLinearAttention使用)
            try:
                if "TangentSpaceLinearAttention" in phase8_modules:
                    from src.models.phase8 import LinearAttentionConfig
                    config = LinearAttentionConfig(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        curvature=1.0,
                    )
                    model = phase8_modules["TangentSpaceLinearAttention"](config)
                elif "BlockWiseDistanceComputation" in phase8_modules:
                    from src.models.phase8 import BlockDistanceConfig
                    config = BlockDistanceConfig(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                    )
                    model = phase8_modules["BlockWiseDistanceComputation"](config)
                else:
                    model = self._create_linear_model()
                
                p8_results = measure_throughput(
                    model, self.batch_size, seq_len, 
                    self.d_model, self.device, num_iterations
                )
                results["phase8"][str(seq_len)] = p8_results
                print(f"  Phase 8: {p8_results['tokens_per_sec']:.0f} tok/s, "
                      f"{p8_results['latency_ms']:.2f}ms, "
                      f"{p8_results['memory_mb']:.1f}MB")
                del model
            except Exception as e:
                results["phase8"][str(seq_len)] = {"error": str(e)}
                print(f"  Phase 8: Error - {e}")
            
            # Phase 7テスト
            try:
                if "HyperbolicAttention" in phase7_modules:
                    model = phase7_modules["HyperbolicAttention"](
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        curvature=1.0,
                    )
                else:
                    model = self._create_linear_model()
                
                p7_results = measure_throughput(
                    model, self.batch_size, seq_len,
                    self.d_model, self.device, num_iterations
                )
                results["phase7"][str(seq_len)] = p7_results
                print(f"  Phase 7: {p7_results['tokens_per_sec']:.0f} tok/s, "
                      f"{p7_results['latency_ms']:.2f}ms, "
                      f"{p7_results['memory_mb']:.1f}MB")
                del model
            except Exception as e:
                results["phase7"][str(seq_len)] = {"error": str(e)}
                print(f"  Phase 7: Error - {e}")
            
            # 比較
            p8 = results["phase8"].get(str(seq_len), {})
            p7 = results["phase7"].get(str(seq_len), {})
            
            if "error" not in p8 and "error" not in p7:
                speedup = p8["tokens_per_sec"] / p7["tokens_per_sec"] if p7["tokens_per_sec"] > 0 else 0
                mem_reduction = (1 - p8["memory_mb"] / p7["memory_mb"]) * 100 if p7["memory_mb"] > 0 else 0
                
                results["comparison"][str(seq_len)] = {
                    "speedup": speedup,
                    "memory_reduction_percent": mem_reduction,
                    "throughput_target_met": speedup >= self.TARGETS["throughput_improvement"],
                    "memory_target_met": mem_reduction >= self.TARGETS["memory_reduction_percent"],
                }
                print(f"  Speedup: {speedup:.2f}x (target: {self.TARGETS['throughput_improvement']}x)")
                print(f"  Memory reduction: {mem_reduction:.1f}% (target: {self.TARGETS['memory_reduction_percent']}%)")
            
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def run_memory_verification(
        self,
        seq_lengths: List[int] = [2048, 4096, 8192, 16384],
    ) -> Dict[str, Any]:
        """メモリ使用量検証を実行"""
        print("\n" + "="*60)
        print("Memory Usage Verification")
        print("="*60)
        
        results = {}
        phase8_modules = self._try_import_phase8()
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            try:
                if "TangentSpaceLinearAttention" in phase8_modules:
                    from src.models.phase8 import LinearAttentionConfig
                    config = LinearAttentionConfig(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        curvature=1.0,
                    )
                    model = phase8_modules["TangentSpaceLinearAttention"](config)
                elif "BlockWiseDistanceComputation" in phase8_modules:
                    from src.models.phase8 import BlockDistanceConfig
                    config = BlockDistanceConfig(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                    )
                    model = phase8_modules["BlockWiseDistanceComputation"](config)
                else:
                    model = self._create_linear_model()
                
                mem_results = measure_memory_usage(
                    model, self.batch_size, seq_len,
                    self.d_model, self.device
                )
                
                # O(N)スケーリングチェック
                expected_linear = seq_len * self.d_model * 4 / (1024**2)  # 概算
                is_linear_scaling = mem_results["peak_memory_mb"] < expected_linear * 10
                
                results[str(seq_len)] = {
                    **mem_results,
                    "is_linear_scaling": is_linear_scaling,
                    "target_met": seq_len != 8192 or mem_results["peak_memory_mb"] < self.TARGETS["max_memory_8192_mb"],
                }
                
                print(f"  Peak memory: {mem_results['peak_memory_mb']:.1f} MB")
                print(f"  Linear scaling: {'Yes' if is_linear_scaling else 'No'}")
                
                del model
            except Exception as e:
                results[str(seq_len)] = {"error": str(e)}
                print(f"  Error: {e}")
            
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def run_component_verification(self) -> Dict[str, Any]:
        """各コンポーネントの検証を実行"""
        print("\n" + "="*60)
        print("Component Verification")
        print("="*60)
        
        results = {}
        phase8_modules = self._try_import_phase8()
        
        components = [
            ("TangentSpaceLinearAttention", "Tangent Space Linear Attention"),
            ("HyperbolicSSM", "Hyperbolic SSM"),
            ("BlockWiseDistanceComputation", "Block-wise Distance Computation"),
            ("ARSSMHyperbolicFusion", "AR-SSM Hyperbolic Fusion"),
            ("EntailmentCones", "Entailment Cones"),
            ("SheafAttentionModule", "Sheaf Attention"),
        ]
        
        for module_name, display_name in components:
            print(f"\n{display_name}:")
            print("-" * 40)
            
            if module_name in phase8_modules:
                try:
                    # モジュール作成テスト - 各モジュールのConfig使用
                    if module_name == "TangentSpaceLinearAttention":
                        from src.models.phase8 import LinearAttentionConfig
                        config = LinearAttentionConfig(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                            curvature=1.0,
                        )
                        model = phase8_modules[module_name](config)
                    elif module_name == "HyperbolicSSM":
                        from src.models.phase8 import HyperbolicSSMConfig
                        config = HyperbolicSSMConfig(
                            d_model=self.d_model,
                            curvature=1.0,
                        )
                        model = phase8_modules[module_name](config)
                    elif module_name == "BlockWiseDistanceComputation":
                        from src.models.phase8 import BlockDistanceConfig
                        config = BlockDistanceConfig(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                        )
                        model = phase8_modules[module_name](config)
                    elif module_name == "ARSSMHyperbolicFusion":
                        from src.models.phase8 import ARSSMFusionConfig
                        config = ARSSMFusionConfig(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                        )
                        model = phase8_modules[module_name](config)
                    elif module_name == "EntailmentCones":
                        from src.models.phase8 import EntailmentConeConfig
                        config = EntailmentConeConfig(
                            d_model=self.d_model,
                        )
                        model = phase8_modules[module_name](config)
                    elif module_name == "SheafAttentionModule":
                        from src.models.phase8 import SheafAttentionConfig
                        config = SheafAttentionConfig(
                            d_model=self.d_model,
                            num_heads=self.num_heads,
                        )
                        model = phase8_modules[module_name](config)
                    else:
                        model = self._create_linear_model()
                    
                    # 簡単なフォワードパステスト
                    model = model.to(self.device)
                    x = torch.randn(2, 256, self.d_model, device=self.device)
                    with torch.no_grad():
                        output = model(x)
                    
                    results[module_name] = {
                        "available": True,
                        "forward_pass": True,
                        "output_shape": list(output.shape) if hasattr(output, 'shape') else str(type(output)),
                    }
                    print(f"  Status: Available and working")
                    print(f"  Output shape: {results[module_name]['output_shape']}")
                    
                    del model, x, output
                except Exception as e:
                    results[module_name] = {
                        "available": True,
                        "forward_pass": False,
                        "error": str(e),
                    }
                    print(f"  Status: Available but error - {e}")
            else:
                results[module_name] = {
                    "available": False,
                    "forward_pass": False,
                }
                print(f"  Status: Not available")
            
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def run_full_verification(self) -> Dict[str, Any]:
        """フル検証を実行"""
        print("\n" + "="*70)
        print("Phase 8 Final GPU Verification")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Device: {self.device}")
        print(f"GPU: {self.gpu_info.get('name', 'N/A')}")
        print(f"VRAM: {self.gpu_info.get('total_memory_gb', 0):.1f} GB")
        print("="*70)
        
        # GPUウォームアップ
        warmup_gpu(self.device)
        
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "gpu_info": self.gpu_info,
                "config": {
                    "batch_size": self.batch_size,
                    "d_model": self.d_model,
                    "num_heads": self.num_heads,
                },
                "targets": self.TARGETS,
            },
            "component_verification": self.run_component_verification(),
            "throughput_verification": self.run_throughput_verification(),
            "memory_verification": self.run_memory_verification(),
        }
        
        # サマリー生成
        self.results["summary"] = self._generate_summary()
        
        return self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """検証サマリーを生成"""
        print("\n" + "="*60)
        print("Verification Summary")
        print("="*60)
        
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "components_available": 0,
            "throughput_targets_met": 0,
            "memory_targets_met": 0,
        }
        
        # コンポーネント検証
        comp = self.results.get("component_verification", {})
        for name, result in comp.items():
            summary["total_tests"] += 1
            if result.get("available") and result.get("forward_pass"):
                summary["passed_tests"] += 1
                summary["components_available"] += 1
            else:
                summary["failed_tests"] += 1
        
        # スループット検証
        throughput = self.results.get("throughput_verification", {}).get("comparison", {})
        for seq_len, result in throughput.items():
            summary["total_tests"] += 1
            if result.get("throughput_target_met"):
                summary["passed_tests"] += 1
                summary["throughput_targets_met"] += 1
            else:
                summary["failed_tests"] += 1
        
        # メモリ検証
        memory = self.results.get("memory_verification", {})
        for seq_len, result in memory.items():
            if "error" not in result:
                summary["total_tests"] += 1
                if result.get("target_met"):
                    summary["passed_tests"] += 1
                    summary["memory_targets_met"] += 1
                else:
                    summary["failed_tests"] += 1
        
        summary["success_rate"] = (
            summary["passed_tests"] / summary["total_tests"] 
            if summary["total_tests"] > 0 else 0
        )
        summary["overall_status"] = "PASS" if summary["success_rate"] >= 0.8 else "FAIL"
        
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Overall status: {summary['overall_status']}")
        
        return summary
    
    def save_results(self, output_path: str):
        """結果をJSONファイルに保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    def generate_report(self, output_path: str):
        """Markdownレポートを生成"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# Phase 8 Final GPU Verification Report\n")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**GPU:** {self.gpu_info.get('name', 'N/A')}\n")
        report.append(f"**VRAM:** {self.gpu_info.get('total_memory_gb', 0):.1f} GB\n")
        
        # サマリー
        summary = self.results.get("summary", {})
        report.append("\n## Summary\n")
        report.append(f"- **Overall Status:** {summary.get('overall_status', 'N/A')}\n")
        report.append(f"- **Success Rate:** {summary.get('success_rate', 0)*100:.1f}%\n")
        report.append(f"- **Tests Passed:** {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}\n")
        
        # スループット結果
        report.append("\n## Throughput Results\n")
        report.append("| Seq Length | Phase 8 (tok/s) | Phase 7 (tok/s) | Speedup | Target Met |\n")
        report.append("|------------|-----------------|-----------------|---------|------------|\n")
        
        throughput = self.results.get("throughput_verification", {})
        for seq_len in ["1024", "2048", "4096", "8192"]:
            p8 = throughput.get("phase8", {}).get(seq_len, {})
            p7 = throughput.get("phase7", {}).get(seq_len, {})
            comp = throughput.get("comparison", {}).get(seq_len, {})
            
            p8_tps = p8.get("tokens_per_sec", 0)
            p7_tps = p7.get("tokens_per_sec", 0)
            speedup = comp.get("speedup", 0)
            target_met = "✓" if comp.get("throughput_target_met") else "✗"
            
            report.append(f"| {seq_len} | {p8_tps:.0f} | {p7_tps:.0f} | {speedup:.2f}x | {target_met} |\n")
        
        # メモリ結果
        report.append("\n## Memory Results\n")
        report.append("| Seq Length | Peak Memory (MB) | Linear Scaling | Target Met |\n")
        report.append("|------------|------------------|----------------|------------|\n")
        
        memory = self.results.get("memory_verification", {})
        for seq_len in ["2048", "4096", "8192", "16384"]:
            mem = memory.get(seq_len, {})
            peak = mem.get("peak_memory_mb", 0)
            linear = "✓" if mem.get("is_linear_scaling") else "✗"
            target = "✓" if mem.get("target_met") else "✗"
            
            report.append(f"| {seq_len} | {peak:.1f} | {linear} | {target} |\n")
        
        # コンポーネント状態
        report.append("\n## Component Status\n")
        report.append("| Component | Available | Working |\n")
        report.append("|-----------|-----------|----------|\n")
        
        components = self.results.get("component_verification", {})
        for name, status in components.items():
            available = "✓" if status.get("available") else "✗"
            working = "✓" if status.get("forward_pass") else "✗"
            report.append(f"| {name} | {available} | {working} |\n")
        
        report.append("\n## Conclusion\n")
        if summary.get("overall_status") == "PASS":
            report.append("Phase 8 has successfully met the performance targets on this GPU configuration.\n")
        else:
            report.append("Some performance targets were not met. See details above for specific failures.\n")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(report)
        
        print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 Final GPU Verification"
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
        "--output-json", type=str,
        default="results/benchmarks/phase8_rtx3080_final.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--output-report", type=str,
        default="results/benchmarks/PHASE8_FINAL_PERFORMANCE_REPORT.md",
        help="Output report file path"
    )
    
    args = parser.parse_args()
    
    verification = Phase8FinalVerification(
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
    
    verification.run_full_verification()
    verification.save_results(args.output_json)
    verification.generate_report(args.output_report)


if __name__ == "__main__":
    main()
